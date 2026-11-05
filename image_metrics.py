"""
Optional per-image metrics (sharpness, entropy, ...) stored on manifest items.

Everything here is pure functions plus a small run-scoped context. No new
dependencies: numpy is always present, torch is imported lazily (and only when
it is the real thing -- the unit tests run against a stub).

Definitions (RGB in [0,1], gamma-encoded as saved). The "analysis image" is the
input with 3 channels, clamped to 0-1, and block-mean downscaled by the smallest
integer factor that brings the long side to <= ANALYSIS_MAX_SIDE, so a 2x
upscale stays comparable with its base and the cost is capped.

  sharpness    variance of the 3x3 Laplacian of luma on the 0-255 scale (valid conv)
  entropy      Shannon entropy (bits, 0..8) of the 256-bin luma histogram
  colorfulness Hasler & Suesstrunk 2003 on 0-255 channels
  contrast     std of luma (RMS contrast, 0..0.5)
  luma         mean of Y = 0.299 R + 0.587 G + 0.114 B
  saturation   mean HSV saturation
  sobel        mean Sobel gradient magnitude of luma (0-1 scale, kernels /4: a full step edge = 1)

These measure signal, not taste: noise raises sharpness and entropy.
ANALYSIS_MAX_SIDE is a versioned constant; changing it breaks comparisons
with metrics stored by older sessions.
"""

import math
import threading
import time

import numpy as np

BUILTIN_METRICS = ("sharpness", "entropy", "colorfulness", "contrast", "luma", "saturation", "sobel")
ANALYSIS_MAX_SIDE = 1024
_SOBEL_SCALE = 0.25

_LUMA_W = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Optional scorer packs (adapters). Each entry is checked against the pack's
# real node signatures (pinned shas below). A disabled entry is reported to the
# Builder UI but never runs and is dropped by normalize_spec(). Keys:
#   loaders[].prefer = {combo_input: [preferred values]} (first one present wins),
#   scale = multiplier applied to the node's output, max_batch = images per call.
# ---------------------------------------------------------------------------
ADAPTERS = {
    # strimmlarn/ComfyUI-Strimmlarns-Aesthetic-Score@42bed3fd78488dda50d83bdc39b5cff6772919c9
    # (LoadAesteticModel: combo model_name; CalculateAestheticScore: image + aesthetic_model,
    # scores image[0] only, returns int(score * 100))
    "aesthetic": {
        "enabled": True,
        "label": "Aesthetic score (LAION, ~1-10)",
        "node_id": "CalculateAestheticScore",
        "image_arg": "image",
        "loaders": [{"node_id": "LoadAesteticModel", "kwargs": {}, "pass_as": "aesthetic_model", "output_index": 0,
                     "prefer": {"model_name": ["sac+logos+ava1-l14-linearMSE.pth", "chadscorer.pth"]}}],
        "output_index": 0,
        "scale": 0.01,
        "max_batch": 1,
        "vram_mb": 1500,
        "pack_hint": "ComfyUI-Strimmlarns-Aesthetic-Score (+ pip pytorch-lightning, openai CLIP; model in models/aesthetic/)",
        "note": "verified against strimmlarn@42bed3f; the node returns score x100 (int), rescaled",
    },
    # ZaneA/ComfyUI-ImageReward@5e9936ded972d8447c26ea2670a92e999036821d
    # (ImageRewardLoader: STRING model; ImageRewardScore: model + prompt + images, FLOAT mean)
    "imagereward": {
        "enabled": False,
        "label": "ImageReward (prompt-aware, ~-2..2)",
        "node_id": "ImageRewardScore",
        "image_arg": "images",
        "text_arg": "prompt",
        "loaders": [{"node_id": "ImageRewardLoader", "kwargs": {"model": "ImageReward-v1.0"}, "pass_as": "model", "output_index": 0}],
        "output_index": 0,
        "max_batch": 1,
        "vram_mb": 2500,
        "pack_hint": "ComfyUI-ImageReward (pip image-reward==1.4; downloads ~1.8 GB on first use)",
        "note": "signatures verified against ZaneA@5e9936d; disabled: image-reward 1.4 does not import under transformers 5 (needs a local med.py patch), see README",
    },
}


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
def normalize_spec(raw):
    """Session `metrics` setting -> {"builtins": [...], "adapters": [...]} or None.

    None means "feature off": missing, not enabled, nothing selected, or garbage.
    Unknown builtin names are dropped; adapters must exist AND be enabled in
    the registry.
    """
    if not isinstance(raw, dict):
        return None
    if "enabled" in raw and not raw.get("enabled"):
        return None
    b = raw.get("builtins") or []
    a = raw.get("adapters") or []
    if isinstance(b, dict):
        b = [k for k, v in b.items() if v]
    if isinstance(a, dict):
        a = [k for k, v in a.items() if v]
    if not isinstance(b, (list, tuple)) or not isinstance(a, (list, tuple)):
        return None
    builtins = [k for k in BUILTIN_METRICS if k in b]
    adapters = [k for k in a if isinstance(k, str) and k in ADAPTERS and ADAPTERS[k].get("enabled")]
    adapters = sorted(set(adapters))
    if not builtins and not adapters:
        return None
    return {"builtins": builtins, "adapters": adapters}


# ---------------------------------------------------------------------------
# numpy reference implementation (also the CPU fallback)
# ---------------------------------------------------------------------------
def _block_factor(h, w):
    side = max(h, w)
    return 1 if side <= ANALYSIS_MAX_SIDE else int(math.ceil(side / ANALYSIS_MAX_SIDE))


def _round4(v):
    return round(float(v), 4)


def _analysis_np(arr):
    """H,W,C (float 0-1 or uint8) -> float32 [H,W,3] clamped, block-mean downscaled."""
    a = np.asarray(arr)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] not in (1, 2, 3, 4) and a.shape[0] in (1, 3, 4):  # CHW
        a = np.transpose(a, (1, 2, 0))
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    a = a[..., :3]
    a = np.clip(a, 0.0, 1.0)  # NaN stays NaN
    h, w = a.shape[:2]
    f = _block_factor(h, w)
    if f > 1:
        hh, ww = (h // f) * f, (w // f) * f
        a = a[:hh, :ww].reshape(hh // f, f, ww // f, f, 3).mean(axis=(1, 3), dtype=np.float32)
    return a


def _var64(x):
    """Population variance with float64 accumulation, float32 inputs (no float64 copy)."""
    m = x.mean(dtype=np.float64)
    d = x - np.float32(m)
    return float((d * d).mean(dtype=np.float64))


def compute_metrics_numpy(arr, keys=BUILTIN_METRICS):
    """Reference implementation. arr: H,W,C float 0-1 or uint8. Returns {key: float}."""
    a = _analysis_np(arr)
    if not np.isfinite(a).all():
        return {}
    keys = [k for k in BUILTIN_METRICS if k in keys]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = _LUMA_W[0] * r + _LUMA_W[1] * g + _LUMA_W[2] * b
    out = {}
    if "sharpness" in keys and y.shape[0] >= 3 and y.shape[1] >= 3:
        y255 = y * np.float32(255.0)
        lap = (y255[:-2, 1:-1] + y255[2:, 1:-1] + y255[1:-1, :-2] + y255[1:-1, 2:] - np.float32(4.0) * y255[1:-1, 1:-1])
        out["sharpness"] = float(_var64(lap))
    if "entropy" in keys:
        q = np.clip(np.round(y * 255.0), 0, 255).astype(np.int64)
        p = np.bincount(q.ravel(), minlength=256).astype(np.float64) / q.size
        p = p[p > 0]
        out["entropy"] = float(-(p * np.log2(p)).sum())
    if "colorfulness" in keys:
        rg = (r - g) * np.float32(255.0)
        yb = (np.float32(0.5) * (r + g) - b) * np.float32(255.0)
        out["colorfulness"] = float(math.sqrt(_var64(rg) + _var64(yb)) +
                                    0.3 * math.sqrt(float(rg.mean(dtype=np.float64)) ** 2 + float(yb.mean(dtype=np.float64)) ** 2))
    if "contrast" in keys:
        out["contrast"] = math.sqrt(_var64(y))
    if "luma" in keys:
        out["luma"] = float(y.mean(dtype=np.float64))
    if "saturation" in keys:
        mx = a.max(axis=-1)
        mn = a.min(axis=-1)
        s = np.where(mx >= 1e-6, (mx - mn) / np.maximum(mx, np.float32(1e-6)), np.float32(0.0))
        out["saturation"] = float(s.mean(dtype=np.float64))
    if "sobel" in keys and y.shape[0] >= 3 and y.shape[1] >= 3:
        two = np.float32(2.0)
        gx = (y[:-2, 2:] + two * y[1:-1, 2:] + y[2:, 2:]) - (y[:-2, :-2] + two * y[1:-1, :-2] + y[2:, :-2])
        gy = (y[2:, :-2] + two * y[2:, 1:-1] + y[2:, 2:]) - (y[:-2, :-2] + two * y[:-2, 1:-1] + y[:-2, 2:])
        out["sobel"] = float((np.sqrt(gx * gx + gy * gy) * np.float32(_SOBEL_SCALE)).mean(dtype=np.float64))
    return {k: _round4(v) for k, v in out.items() if math.isfinite(v)}


# ---------------------------------------------------------------------------
# torch implementation (GPU, one sync per batch)
# ---------------------------------------------------------------------------
def _real_torch():
    try:
        import torch
    except Exception:
        return None
    return torch if hasattr(torch, "from_numpy") and hasattr(torch, "conv2d") else None


def _is_tensor(x):
    t = _real_torch()
    return t is not None and t.is_tensor(x)


def _to_bchw(images):
    """Any [B,H,W,C] / [B,C,H,W] / [H,W,C] / 5-D tensor -> float32 [B,3,H,W] clamped 0-1."""
    torch = _real_torch()
    x = images.detach().float()
    if x.dim() == 2:
        x = x[..., None]
    if x.dim() == 3:
        x = x[None]
    if x.dim() > 4:
        x = x.reshape(-1, *x.shape[-3:])
    if x.shape[-1] in (1, 3, 4):      # BHWC (ComfyUI IMAGE convention)
        x = x.permute(0, 3, 1, 2)
    elif x.shape[1] not in (1, 3, 4):
        raise ValueError(f"unsupported image tensor shape {tuple(images.shape)}")
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    x = x[:, :3].clamp(0.0, 1.0)
    h, w = x.shape[-2:]
    f = _block_factor(h, w)
    if f > 1:
        x = torch.nn.functional.avg_pool2d(x, f)   # floor mode == crop to a multiple, then block mean
    return x.contiguous()


def compute_metrics_torch(images, keys=BUILTIN_METRICS):
    """Per-image metrics for a batch tensor. Returns list[dict] (one per image).

    Runs under inference_mode; a single .cpu().tolist() syncs the device once.
    """
    torch = _real_torch()
    if torch is None:
        raise RuntimeError("torch is not available")
    with torch.inference_mode():
        keys = [k for k in BUILTIN_METRICS if k in keys]
        x = _to_bchw(images)
        B, _, H, W = x.shape
        finite = torch.isfinite(x).reshape(B, -1).all(dim=1)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = _LUMA_W[0] * r + _LUMA_W[1] * g + _LUMA_W[2] * b      # [B,H,W]
        cols = []
        names = []
        if "sharpness" in keys and H >= 3 and W >= 3:
            y255 = (y * 255.0)[:, None]
            kern = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                                device=y.device, dtype=y.dtype)[None, None]
            lap = torch.nn.functional.conv2d(y255, kern)
            cols.append(lap.reshape(B, -1).var(dim=1, unbiased=False))
            names.append("sharpness")
        if "entropy" in keys:
            q = torch.clamp(torch.round(y * 255.0), 0, 255).long().reshape(B, -1)
            q = q + (torch.arange(B, device=q.device) * 256)[:, None]
            counts = torch.bincount(q.reshape(-1), minlength=256 * B).reshape(B, 256).float()
            p = counts / float(H * W)
            ent = -torch.where(p > 0, p * torch.log2(p.clamp_min(1e-30)), torch.zeros_like(p)).sum(dim=1)
            cols.append(ent)
            names.append("entropy")
        if "colorfulness" in keys:
            rg = ((r - g) * 255.0).reshape(B, -1)
            yb = ((0.5 * (r + g) - b) * 255.0).reshape(B, -1)
            std = torch.sqrt(rg.var(dim=1, unbiased=False) + yb.var(dim=1, unbiased=False))
            mean = torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2)
            cols.append(std + 0.3 * mean)
            names.append("colorfulness")
        if "contrast" in keys:
            cols.append(y.reshape(B, -1).var(dim=1, unbiased=False).sqrt())
            names.append("contrast")
        if "luma" in keys:
            cols.append(y.reshape(B, -1).mean(dim=1))
            names.append("luma")
        if "saturation" in keys:
            mx = x.max(dim=1).values
            mn = x.min(dim=1).values
            s = torch.where(mx >= 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
            cols.append(s.reshape(B, -1).mean(dim=1))
            names.append("saturation")
        if "sobel" in keys and H >= 3 and W >= 3:
            kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=y.device, dtype=y.dtype)
            k = (torch.stack([kx, kx.t()]) * _SOBEL_SCALE)[:, None]        # [2,1,3,3]
            gr = torch.nn.functional.conv2d(y[:, None], k)                   # [B,2,H-2,W-2]
            cols.append(gr.pow(2).sum(dim=1).sqrt().reshape(B, -1).mean(dim=1))
            names.append("sobel")
        if not cols:
            return [{} for _ in range(B)]
        table = torch.stack([c.float() for c in cols], dim=1).cpu().tolist()   # the one sync
        ok = finite.cpu().tolist()
    out = []
    for i in range(B):
        d = {}
        if ok[i]:
            for n, v in zip(names, table[i]):
                if math.isfinite(v):
                    d[n] = _round4(v)
        out.append(d)
    return out


def _comfy_device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return None


def compute_for_pil(pil, keys=BUILTIN_METRICS, use_torch=True):
    """PIL image -> metrics. Torch on the comfy device when torch is real, else numpy."""
    arr = np.asarray(pil.convert("RGB"))
    torch = _real_torch() if use_torch else None
    if torch is not None:
        try:
            t = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)[None]
            dev = _comfy_device()
            if dev is not None:
                t = t.to(dev)
            return compute_metrics_torch(t, keys)[0]
        except Exception:
            pass
    return compute_metrics_numpy(arr, keys)


# ---------------------------------------------------------------------------
# Run-scoped context
# ---------------------------------------------------------------------------
class _RunContext:
    def __init__(self, spec):
        self.spec = spec
        self.warned = set()
        self.loader_cache = {}       # adapter name -> {"outputs": {...}} (loader outputs live for the run)
        self.disabled_adapters = set()
        self.checked_adapters = set()
        self.times = []              # seconds per image (builtins only)
        self.adapter_times = {}      # adapter name -> [seconds per call]
        self.lock = threading.Lock()
        self.closed = False


_ctx_lock = threading.Lock()
_stack = []       # contexts of runs still open; the newest is the "current" one
_CURRENT = object()   # sentinel: "use the active run's spec"


def begin_run(spec):
    """Start a run context and return its token. spec: normalized dict or raw setting or None.

    The caller passes the token back to end_run(token) so a run can only close
    its own context: end_run never clobbers a context another run started
    meanwhile. The dashboard upscale thread does not use this context at all
    (it passes an explicit spec to attach()).
    """
    ctx = _RunContext(normalize_spec(spec))
    with _ctx_lock:
        _stack.append(ctx)
    return ctx


def end_run(token=None):
    """Close a run: print the timing line, release adapter loader outputs.

    With a token only that context is closed (and `current` cleared only if it
    still points at it). Without a token the active context is closed.
    """
    with _ctx_lock:
        ctx = token if token is not None else (_stack[-1] if _stack else None)
        if ctx is None or ctx.closed:
            return
        ctx.closed = True
        if ctx in _stack:
            _stack.remove(ctx)
    with ctx.lock:
        times = list(ctx.times)
        ctx.loader_cache.clear()
        adapter_times = {k: list(v) for k, v in ctx.adapter_times.items()}
    if adapter_times:
        _release_vram(None)
    for aname, ts in adapter_times.items():
        if ts:
            print(f"[GridTester] 📏 metric adapter '{aname}': {len(ts)} images, mean {sum(ts) * 1000.0 / len(ts):.0f} ms")
    if times:
        ms = [t * 1000.0 for t in times]
        print(f"[GridTester] 📏 metrics: {len(ms)} images, mean {sum(ms) / len(ms):.1f} ms, max {max(ms):.1f} ms")


def current_spec():
    ctx = _active_ctx()
    return ctx.spec if ctx is not None else None


def _active_ctx():
    with _ctx_lock:
        return _stack[-1] if _stack else None


def _warn_once(ctx, key, msg):
    if ctx is not None:
        with ctx.lock:
            if key in ctx.warned:
                return
            ctx.warned.add(key)
    print(f"[GridTester] ⚠️ {msg}")


def record_time(seconds, ctx=None):
    ctx = ctx or _active_ctx()
    if ctx is not None:
        with ctx.lock:
            ctx.times.append(seconds)


def sink_update(sink, decoded, spec=_CURRENT):
    """Hook for decode_latent_with_vae: compute builtin metrics on the GPU tensor into `sink`.

    spec defaults to the active run's; pass an explicit spec (or None = off)
    from threads that must not read generation-run state. Never raises.
    """
    ctx = None
    if spec is _CURRENT:
        ctx = _active_ctx()
        spec = ctx.spec if ctx else None
    if not spec or not spec.get("builtins"):
        return
    try:
        t0 = time.perf_counter()
        sink.update(compute_metrics_torch(decoded, spec["builtins"])[0])
        record_time(time.perf_counter() - t0, ctx)
    except Exception as e:
        _warn_once(ctx, "skip", f"metrics skipped: {e}")


# ---------------------------------------------------------------------------
# attach (never raises)
# ---------------------------------------------------------------------------
def attach(meta, image, spec=_CURRENT, builtins=True, adapters=True):
    """Set meta["metrics"] from `image` (torch tensor | PIL | numpy). Never raises.

    spec: omitted -> the active run's spec; explicit dict/None -> exactly that
    (None = do nothing). Explicit specs run builtins only (adapters need a run
    context for their loader cache).
    """
    ctx = None
    explicit = spec is not _CURRENT
    if not explicit:
        ctx = _active_ctx()
        spec = ctx.spec if ctx else None
    if not spec:
        return
    try:
        metrics = {}
        bkeys = spec.get("builtins") or []
        if builtins and bkeys:
            t0 = time.perf_counter()
            if _is_tensor(image):
                metrics.update(compute_metrics_torch(image, bkeys)[0])
            elif isinstance(image, np.ndarray):
                metrics.update(compute_metrics_numpy(image, bkeys))
            else:
                metrics.update(compute_for_pil(image, bkeys))
            if ctx is not None:
                record_time(time.perf_counter() - t0, ctx)
        if adapters and ctx is not None and spec.get("adapters"):
            metrics.update(_run_adapters(ctx, spec["adapters"], image, meta))
        if metrics:
            meta["metrics"] = metrics      # replaces: a derived image never inherits its source's metrics
    except Exception as e:
        _warn_once(ctx, "skip", f"metrics skipped: {e}")


def spec_from_item(item):
    """Builtin-only spec from a source item's stored metrics (None when it has none).

    Used by the dashboard upscale thread, which has no session settings and must
    not read generation-run state: an upscale of a measured image is measured the
    same way, an upscale of an unmeasured image is not.
    """
    m = item.get("metrics") if isinstance(item, dict) else None
    if not isinstance(m, dict):
        return None
    keys = [k for k in BUILTIN_METRICS if k in m]
    return {"builtins": keys, "adapters": []} if keys else None


def finalize(meta, image, sink=None, spec=_CURRENT):
    """Metrics for a newly saved image. `sink` = builtins already computed at decode time.

    Always drops any metrics copied from a source item first (never inherit).
    spec omitted -> the active run's context; explicit spec/None -> exactly that
    (never touches generation-run state). Never raises.
    """
    meta.pop("metrics", None)
    explicit = spec is not _CURRENT
    if sink:
        meta["metrics"] = dict(sink)
        if not explicit:
            attach_adapters(meta, image)
    elif image is not None:
        attach(meta, image, spec=spec)


def attach_adapters(meta, image):
    """Run only the active run's adapters (builtins already came from the decode sink)."""
    ctx = _active_ctx()
    if ctx is None or not ctx.spec or not ctx.spec.get("adapters"):
        return
    try:
        res = _run_adapters(ctx, ctx.spec["adapters"], image, meta)
        if res:
            meta.setdefault("metrics", {}).update(res)
    except Exception as e:
        _warn_once(ctx, "skip", f"metrics skipped: {e}")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
def _node_mappings():
    try:
        import nodes
        return getattr(nodes, "NODE_CLASS_MAPPINGS", None) or {}
    except Exception:
        return {}


def _is_v3(cls):
    try:
        from comfy_api.latest import io as _io
        return isinstance(cls, type) and issubclass(cls, _io.ComfyNode)
    except Exception:
        return False


def _call_node(cls, kwargs):
    """Same dispatch as image_generation._apply_model_patch_node."""
    if _is_v3(cls):
        result = cls.execute(**kwargs)
    else:
        result = getattr(cls(), getattr(cls, "FUNCTION", "execute"))(**kwargs)
    return result


def _output(result, index):
    if hasattr(result, "args") and isinstance(getattr(result, "args"), tuple):
        return result.args[index]
    if isinstance(result, (tuple, list)):
        return result[index]
    return result


def _required_inputs(cls):
    """{name: default-or-_MISSING} for required inputs, or None when it cannot be introspected."""
    _MISSING = object()
    try:
        if _is_v3(cls):
            schema = cls.GET_SCHEMA() if hasattr(cls, "GET_SCHEMA") else cls.define_schema()
            out = {}
            for i in schema.inputs:
                if getattr(i, "optional", False):
                    continue
                d = getattr(i, "default", None)
                out[i.id] = _MISSING if d is None else d
            return out, _MISSING
        spec = cls.INPUT_TYPES().get("required", {})
        out = {}
        for name, v in spec.items():
            kind = v[0] if isinstance(v, (tuple, list)) and v else v
            opts = v[1] if isinstance(v, (tuple, list)) and len(v) > 1 and isinstance(v[1], dict) else {}
            if "default" in opts:
                out[name] = opts["default"]
            elif isinstance(kind, (list, tuple)) and kind:
                out[name] = kind[0]            # combo: first option
            else:
                out[name] = _MISSING
        return out, _MISSING
    except Exception:
        return None


def _adapter_defs(name):
    ad = ADAPTERS.get(name)
    return ad if ad and ad.get("enabled") else None


def _combo_options(cls, name):
    """Option list of a V1 combo input (required or optional), or None when `name` is not a combo."""
    try:
        types = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            v = types.get(section, {}).get(name)
            if v is not None:
                kind = v[0] if isinstance(v, (tuple, list)) and v else v
                if isinstance(kind, (list, tuple)):
                    return list(kind)
                return None
    except Exception:
        pass
    return None


def _loader_kwargs(cls, loader):
    """Kwargs for a loader node: required-input defaults (combo -> first option), then the
    `prefer` picks (first preferred value present in the combo), then explicit `kwargs`.

    Returns (kwargs, problem). problem is a short reason when a required input stays uncovered
    or a combo has no options at all (e.g. no model file in the pack's folder).
    """
    kwargs = {}
    req = _required_inputs(cls)
    missing = None
    if req is not None:
        req, missing = req
    explicit = dict(loader.get("kwargs", {}))
    prefer = loader.get("prefer", {}) or {}
    for name, default in (req or {}).items():
        opts = _combo_options(cls, name)
        if opts is not None and not opts and name not in explicit:
            if loader.get("node_id") == "LoadAesteticModel":
                return kwargs, "no model file in models/aesthetic (download sac+logos+ava1-l14-linearMSE.pth)"
            return kwargs, f"'{loader['node_id']}' input '{name}' has no options (no model file installed?)"
        if default is not missing:
            kwargs[name] = default
    for name, wanted in prefer.items():
        opts = _combo_options(cls, name) or []
        for w in wanted:
            if w in opts:
                kwargs[name] = w
                break
    kwargs.update(explicit)
    if req is not None:
        for name, default in req.items():
            if name not in kwargs and default is missing:
                return kwargs, f"required input '{name}' of '{loader['node_id']}' is not covered"
    return kwargs, None


def _adapter_problem(ad, mappings):
    """Return None when installed and call-compatible, else a short reason."""
    ids = [ad["node_id"]] + [l["node_id"] for l in ad.get("loaders", [])]
    for nid in ids:
        if nid not in mappings:
            return f"node '{nid}' not installed"
    for loader in ad.get("loaders", []):
        _, problem = _loader_kwargs(mappings[loader["node_id"]], loader)
        if problem:
            return problem
    covered = {ad.get("image_arg"), ad.get("text_arg")} | set(ad.get("kwargs", {}))
    covered |= {l["pass_as"] for l in ad.get("loaders", [])}
    covered.discard(None)
    req = _required_inputs(mappings[ad["node_id"]])
    if req is not None:
        req, missing = req
        for name, default in req.items():
            if name not in covered and default is missing:
                return f"required input '{name}' of '{ad['node_id']}' is not covered"
    return None


def adapter_status():
    """For the Builder hint route: [{name,label,installed,enabled,problem,pack_hint,note}]."""
    mappings = _node_mappings()
    out = []
    for name, ad in ADAPTERS.items():
        installed = all(n in mappings for n in [ad["node_id"]] + [l["node_id"] for l in ad.get("loaders", [])])
        try:
            problem = _adapter_problem(ad, mappings)
        except Exception as e:
            problem = str(e)
        out.append({
            "name": name,
            "label": ad.get("label", name),
            "enabled": bool(ad.get("enabled")),
            "installed": installed,
            "problem": problem,
            "pack_hint": ad.get("pack_hint", ""),
            "note": ad.get("note", ""),
        })
    return out


def _release_vram(need_mb):
    """Ask ComfyUI's memory manager to make room (need_mb) or just empty the cache (None). Never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        import comfy.model_management as mm
        if need_mb:
            mm.free_memory(int(need_mb) * 1024 * 1024, mm.get_torch_device())
        else:
            import gc
            gc.collect()
            mm.soft_empty_cache()
    except Exception:
        pass


def _adapter_image(image):
    """Whatever we were given -> [B,H,W,3] float tensor in [0,1] when torch is real (else numpy)."""
    torch = _real_torch()
    if _is_tensor(image):
        return _to_bhwc_full(image).cpu()      # some packs call .numpy() on the input
    arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    arr = arr[None] if arr.ndim == 3 else arr
    return torch.from_numpy(np.ascontiguousarray(arr)) if torch is not None else arr


def _to_bhwc_full(t):
    """Full-resolution (no analysis downscale) BHWC float view of a tensor."""
    x = t.detach().float()
    if x.dim() == 3:
        x = x[None]
    if x.dim() > 4:
        x = x.reshape(-1, *x.shape[-3:])
    if x.shape[-1] not in (1, 3, 4) and x.shape[1] in (1, 3, 4):
        x = x.permute(0, 2, 3, 1)
    return x[..., :3].clamp(0.0, 1.0).contiguous()


def _float_of(v):
    if hasattr(v, "item") and not isinstance(v, (list, tuple, str)):
        try:
            return float(v.reshape(-1)[0].item()) if hasattr(v, "reshape") else float(v.item())
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return _float_of(v[0])
    return float(v)


def _run_adapters(ctx, names, image, meta):
    mappings = _node_mappings()
    out = {}
    img = None
    for name in names:
        ad = _adapter_defs(name)
        if ad is None or name in ctx.disabled_adapters:
            continue
        try:
            if name not in ctx.checked_adapters:
                ctx.checked_adapters.add(name)
                problem = _adapter_problem(ad, mappings)
                if problem:
                    raise RuntimeError(problem)
            if img is None:
                img = _adapter_image(image)
            t0 = time.perf_counter()
            outputs = ctx.loader_cache.setdefault(name, {})
            kwargs = dict(ad.get("kwargs", {}))
            for li, loader in enumerate(ad.get("loaders", [])):
                if li not in outputs:
                    lk, lproblem = _loader_kwargs(mappings[loader["node_id"]], loader)
                    if lproblem:
                        raise RuntimeError(lproblem)
                    res = _call_node(mappings[loader["node_id"]], lk)
                    outputs[li] = _output(res, loader.get("output_index", 0))
                kwargs[loader["pass_as"]] = outputs[li]
            mb = ad.get("max_batch")
            kwargs[ad["image_arg"]] = img[:mb] if mb and getattr(img, "shape", None) is not None and img.shape[0] > mb else img
            if ad.get("text_arg"):
                _txt = str(meta.get("positive") or meta.get("config_positive") or "")
                try:
                    from .regional_prompt import plain_prompt_text   # COUPLE: score the plain text, not the markers
                    _txt = plain_prompt_text(_txt)
                except ImportError:
                    pass
                kwargs[ad["text_arg"]] = _txt
            cls = mappings[ad["node_id"]]
            req = _required_inputs(cls)
            if req is not None:
                req, missing = req
                for k, d in req.items():
                    if k not in kwargs and d is not missing:
                        kwargs[k] = d          # widget defaults
            _release_vram(ad.get("vram_mb"))
            res = _call_node(cls, kwargs)
            out[name] = _round4(_float_of(_output(res, ad.get("output_index", 0))) * ad.get("scale", 1.0))
            _release_vram(None)
            with ctx.lock:
                ctx.adapter_times.setdefault(name, []).append(time.perf_counter() - t0)
        except Exception as e:
            ctx.disabled_adapters.add(name)
            _warn_once(ctx, f"adapter:{name}", f"metric adapter '{name}' disabled: {e}")
    return out
