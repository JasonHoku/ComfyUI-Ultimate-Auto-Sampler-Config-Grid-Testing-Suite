"""Florence2 Hi-Res Fix — targeted-region inpaint via Florence2 referring-expression segmentation.

Soft dependency: kijai/ComfyUI-Florence2 (checked at job start, not at import).
Required ComfyUI built-ins: GrowMask, FeatherMask (always present).
Crop / uncrop / resize-back are pure Python (no kjnodes dep).

Pipeline (per image, per Florence2 step):
  1. Pre-flight (once per job, cached)
  2. Load Florence2 model (cached per job)
  3. Detect via Florence2Run(task=referring_expression_segmentation)
  4. GrowMask
  5. Crop by mask (pure Python)
  6. Megapixel resize
  7. VAE encode -> generate_image (project's known-working sampling path)
  8. VAE decode -> resize back to crop dims
  9. FeatherMask
 10. Paste back (pure Python)
 11. Save + manifest entry (handled by upscale_runner)
"""

import builtins
import math
import sys


def safe_print(*args, **kwargs):
    """Windows-safe print mirroring config_builder_node.safe_print."""
    try:
        builtins.print(*args, **kwargs)
    except (OSError, ValueError):
        try:
            msg = " ".join(str(a) for a in args) + kwargs.get("end", "\n")
            sys.__stdout__.write(msg)
            sys.__stdout__.flush()
        except Exception:
            pass


print = safe_print


def _vram_log(label):
    """Diagnostic VRAM probe. Prints current CUDA allocated + reserved memory.
    Used to characterize the per-image VRAM curve of Florence2 Hi-Res Fix —
    pinpoints whether OOMs come from a cumulative leak across the batch or a
    one-shot peak on a runaway image. Safe no-op if torch isn't available or
    CUDA isn't present (CPU-only test runs).
    """
    try:
        import torch as _t
        if _t.cuda.is_available():
            alloc = _t.cuda.memory_allocated() / 1024**3
            reserved = _t.cuda.memory_reserved() / 1024**3
            safe_print(f"[Florence2HiResFix VRAM] {label}: alloc={alloc:.2f}GB reserved={reserved:.2f}GB")
    except Exception:
        pass


def compute_target_dims(src_w, src_h, target_megapixels):
    """Compute target (width, height) for a megapixel-based resize.

    Uses mebipixel convention (1 MP = 1024 * 1024 pixels), matching ComfyUI core's
    ImageScaleToTotalPixels. Output dims are snapped to multiples of 8, floored at 64,
    capped at 4096.

    Args:
        src_w: source width in pixels (int)
        src_h: source height in pixels (int)
        target_megapixels: target area in MP (float, e.g. 1.0 = ~1024x1024)

    Returns:
        Tuple (target_w, target_h) of ints, both divisible by 8.
    """
    target_pixels = target_megapixels * 1024 * 1024
    src_pixels = src_w * src_h
    if src_pixels <= 0:
        raise ValueError(f"Source dimensions invalid: {src_w}x{src_h}")
    scale = math.sqrt(target_pixels / src_pixels)
    new_w = max(64, min(4096, int(round(src_w * scale)) // 8 * 8))
    new_h = max(64, min(4096, int(round(src_h * scale)) // 8 * 8))
    return new_w, new_h


def parse_mask_select_indices(user_input, detected_count):
    """Parse user's output_mask_select string against detected region count.

    Args:
        user_input: string from UI; "" = use all, "0" = first, "0,2" = specific indices.
        detected_count: how many regions Florence2 actually returned.

    Returns:
        Tuple (indices: list[int], mode: str) where mode is one of:
        - "all": use all detected regions (caller should union)
        - "select": use the listed indices
        - "no_detection": no valid selection possible (caller should treat as miss)
    """
    if detected_count <= 0:
        return [], "no_detection"

    stripped = (user_input or "").strip()
    if not stripped:
        return [], "all"

    raw_tokens = [t.strip() for t in stripped.split(",") if t.strip()]
    if not raw_tokens:
        return [], "all"

    parsed = []
    for tok in raw_tokens:
        try:
            n = int(tok)
        except ValueError:
            return [], "no_detection"
        if n < 0:
            return [], "no_detection"
        if n < detected_count:
            parsed.append(n)
        # OOR indices silently dropped

    if not parsed:
        return [], "no_detection"

    # Stable dedupe preserving order
    seen = set()
    out = []
    for n in parsed:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, "select"


def _crop_image_by_mask(image, mask, padding, min_crop_resolution, max_crop_resolution):
    """Crop image to the bbox of a mask, with padding and min/max constraints.

    Args:
        image: torch tensor (B, H, W, C) float32 in [0,1]
        mask: torch tensor (B, H, W) float32 in [0,1]
        padding: int, pixels to expand bbox each side before clamping
        min_crop_resolution: int, minimum bbox dim — expands bbox if smaller
        max_crop_resolution: int, maximum bbox dim — shrinks bbox if larger

    Returns:
        Tuple (cropped_image, cropped_mask, bbox) where bbox = (x0, y0, w, h).

    Raises:
        ValueError: mask is empty (sum == 0).
    """
    import torch

    # Caller is responsible for no-detection check; this is defense-in-depth.
    if mask.sum() <= 0:
        raise ValueError("Mask is empty — cannot compute bbox")

    # Find tight bbox via nonzero indices on a 2D view of the first batch.
    # Mask is (B, H, W); we use batch 0 (Florence2Run returns single-batch).
    m2d = mask[0]  # (H, W)
    img_h, img_w = m2d.shape

    nz = torch.nonzero(m2d > 0.5)  # (N, 2) = (y, x)
    if nz.numel() == 0:
        raise ValueError("Mask is empty after thresholding")

    y_min = int(nz[:, 0].min().item())
    y_max = int(nz[:, 0].max().item())
    x_min = int(nz[:, 1].min().item())
    x_max = int(nz[:, 1].max().item())

    # Apply padding
    x0 = max(0, x_min - padding)
    y0 = max(0, y_min - padding)
    x1 = min(img_w, x_max + 1 + padding)
    y1 = min(img_h, y_max + 1 + padding)

    bw = x1 - x0
    bh = y1 - y0

    # Apply min_crop_resolution by expanding around center, clamping to image bounds.
    if bw < min_crop_resolution:
        cx = (x0 + x1) // 2
        half = min_crop_resolution // 2
        x0 = max(0, cx - half)
        x1 = min(img_w, x0 + min_crop_resolution)
        x0 = max(0, x1 - min_crop_resolution)  # second pass in case clamp shifted
        bw = x1 - x0
    if bh < min_crop_resolution:
        cy = (y0 + y1) // 2
        half = min_crop_resolution // 2
        y0 = max(0, cy - half)
        y1 = min(img_h, y0 + min_crop_resolution)
        y0 = max(0, y1 - min_crop_resolution)
        bh = y1 - y0

    # Apply max_crop_resolution by contracting around center.
    if bw > max_crop_resolution:
        cx = (x0 + x1) // 2
        half = max_crop_resolution // 2
        x0 = cx - half
        x1 = x0 + max_crop_resolution
        bw = x1 - x0
    if bh > max_crop_resolution:
        cy = (y0 + y1) // 2
        half = max_crop_resolution // 2
        y0 = cy - half
        y1 = y0 + max_crop_resolution
        bh = y1 - y0

    # Final clamp to image (in case max shrinking pushed us off-bounds)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    bw = x1 - x0
    bh = y1 - y0

    cropped_image = image[:, y0:y1, x0:x1, :].contiguous()
    cropped_mask = mask[:, y0:y1, x0:x1].contiguous()
    bbox = (x0, y0, bw, bh)
    return cropped_image, cropped_mask, bbox


def _paste_into_image(destination, source, mask, bbox):
    """Paste a cropped image back into the destination using mask for alpha blending.

    Args:
        destination: torch tensor (B, H, W, C) — the full original image to paste into
        source: torch tensor (B, h, w, C) — the cropped (and resized) image to paste
        mask: torch tensor (B, h, w) — alpha mask for the source (0..1)
        bbox: tuple (x0, y0, w, h) — where in destination to paste

    Returns:
        New torch tensor (B, H, W, C) with the blended result.
    """
    x0, y0, bw, bh = bbox
    src_h, src_w = source.shape[1], source.shape[2]

    # If source dims don't match bbox (caller error), use source dims
    if src_h != bh or src_w != bw:
        bh = src_h
        bw = src_w

    result = destination.clone()
    # Broadcast (B, h, w) mask to (B, h, w, C)
    mask_3c = mask.unsqueeze(-1).expand(-1, -1, -1, source.shape[-1])

    dest_slice = destination[:, y0:y0 + bh, x0:x0 + bw, :]
    blended = source * mask_3c + dest_slice * (1.0 - mask_3c)
    result[:, y0:y0 + bh, x0:x0 + bw, :] = blended
    return result


# Required Florence2 node class names (looked up via nodes.NODE_CLASS_MAPPINGS)
REQUIRED_FLORENCE2_NODE_NAMES = [
    "Florence2Run",
    "DownloadAndLoadFlorence2Model",
]


def get_florence2_node_classes():
    """Look up Florence2 nodes in NODE_CLASS_MAPPINGS.

    Returns:
        Dict mapping node name to class.

    Raises:
        RuntimeError: any required node is missing.
    """
    import nodes
    found = {}
    missing = []
    for name in REQUIRED_FLORENCE2_NODE_NAMES:
        cls = nodes.NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            missing.append(name)
        else:
            found[name] = cls
    if missing:
        raise RuntimeError(
            "Florence2 Hi-Res Fix requires the kijai/ComfyUI-Florence2 custom node.\n"
            "Missing node(s): " + ", ".join(missing) + "\n\n"
            "Install via ComfyUI Manager:\n"
            "  search 'Florence-2' -> install -> restart Comfy\n\n"
            "Or manually:\n"
            "  cd ComfyUI/custom_nodes\n"
            "  git clone https://github.com/kijai/ComfyUI-Florence2\n"
            "  restart Comfy"
        )
    return found


def preflight_florence2():
    """Validate the Florence2 dependencies are installed.

    Call ONCE per job before any image is processed. Raises with a clear
    install hint if either required node is missing.
    """
    get_florence2_node_classes()


# Module-level cache for Florence2 models. Keyed by model name string.
# Cleared on Comfy restart; survives across jobs in the same session.
_FLORENCE2_MODEL_CACHE = {}


def load_florence2_model(model_name):
    """Load a Florence2 model, caching by name across jobs.

    Args:
        model_name: HF Hub id, e.g. "microsoft/Florence-2-base"

    Returns:
        The loaded model handle (whatever the loader node returns at index 0).
    """
    if model_name in _FLORENCE2_MODEL_CACHE:
        return _FLORENCE2_MODEL_CACHE[model_name]

    try:
        from .ltx_video_generation import _call_node, _unwrap
    except ImportError:
        from ltx_video_generation import _call_node, _unwrap

    classes = get_florence2_node_classes()
    loader_cls = classes["DownloadAndLoadFlorence2Model"]

    print(f"[Florence2HiResFix] Loading {model_name}...")
    result = _call_node(
        loader_cls,
        model=model_name,
        precision="fp16",
        attention="sdpa",
        convert_to_safetensors=False,
    )
    handle = _unwrap(result, 0)
    _FLORENCE2_MODEL_CACHE[model_name] = handle
    return handle


# Indirection so tests can monkeypatch without importing model_loader (which
# would pull in torch/comfy and slow the test boot). At runtime these resolve
# to the real loaders inside _get_or_load_checkpoint_lora.
_FH_CKPT_LOADER = None
_FH_LORA_LOADER = None


def _resolve_loaders():
    """Lazy-import the real loaders. Only called from runtime path; tests
    monkeypatch _FH_CKPT_LOADER / _FH_LORA_LOADER directly.

    Uses load_loras_for_preencoding (NOT load_loras) — its signature is
    (base_model, base_clip, lora_string) and returns (model, clip), exactly
    matching what we need. The full load_loras() requires target_model_name
    and incompatible_loras dict tracking which is only meaningful when
    iterating across multiple grid configs."""
    global _FH_CKPT_LOADER, _FH_LORA_LOADER
    if _FH_CKPT_LOADER is None or _FH_LORA_LOADER is None:
        try:
            from .model_loader import load_checkpoint, load_loras_for_preencoding
        except ImportError:
            from model_loader import load_checkpoint, load_loras_for_preencoding
        _FH_CKPT_LOADER = _FH_CKPT_LOADER or load_checkpoint
        _FH_LORA_LOADER = _FH_LORA_LOADER or load_loras_for_preencoding
    return _FH_CKPT_LOADER, _FH_LORA_LOADER


def _get_or_load_checkpoint_lora(item, fallback, model_source, cache, session_model_name=None):
    """Resolve (model, clip, vae) handles for one manifest item, honoring model_source.

    VRAM optimization: when the item's model matches the session's already-loaded
    model (same `session_model_name`), we reuse the session's base weights and just
    apply the item's LoRAs on top via `load_loras_for_preencoding`. This avoids
    holding TWO full copies of the same checkpoint in VRAM (one loaded by
    upscale_runner, one freshly loaded here). On 8 GB cards this duplication was
    enough to OOM during Florence2's beam search.

    Args:
        item: manifest entry dict — may have 'model', 'lora_expanded' keys
        fallback: tuple (model, clip, vae) — the already-loaded session-default handles
        model_source: "from_manifest" or "from_builder"
        cache: dict — job-local cache keyed by (model_name, lora_string)
        session_model_name: name of the model upscale_runner already loaded (for the
            same-model short-circuit). Optional; if None, every from_manifest item
            does a fresh load.

    Returns:
        Tuple (model, clip, vae).
    """
    if model_source != "from_manifest":
        return fallback

    model_name = item.get("model", "").strip()
    if not model_name:
        # Legacy entry without model key — fall back, but only log once per job.
        if not cache.get("_warned_missing_model"):
            print("[Florence2HiResFix] Manifest item missing 'model' field; using session default")
            cache["_warned_missing_model"] = True
        return fallback

    # Read lora_expanded (post-folder-expansion concrete file paths) with fallback to
    # the original 'lora' field for legacy manifests written before lora_expanded was
    # tracked. Both formats are accepted by parse_lora_definition.
    # Also normalize: strip whitespace, collapse any trailing " + " (which can leak
    # from " + ".join(parts) where parts has empty trailing entries from expand_lora_folder).
    raw_lora = item.get("lora_expanded") or item.get("lora") or ""
    lora_string = str(raw_lora).strip()
    # Strip trailing "+", " +", "+ ", " + " — these break parse_lora_definition's
    # float parsing on the last entry (float("0.20 +") -> ValueError).
    while lora_string.endswith("+") or lora_string.endswith(" "):
        lora_string = lora_string.rstrip("+ ").rstrip()
    # Same for leading.
    while lora_string.startswith("+") or lora_string.startswith(" "):
        lora_string = lora_string.lstrip("+ ").lstrip()
    if not lora_string:
        lora_string = "None"
    key = (model_name, lora_string)
    if key in cache:
        return cache[key]

    ckpt_fn, lora_fn = _resolve_loaders()

    # Same-model short-circuit: reuse session's base, just apply the item's LoRAs.
    # This saves ~2-7 GiB depending on checkpoint size (SD1.5 ~2GB, SDXL ~7GB).
    if session_model_name and model_name == session_model_name:
        base_model, base_clip, base_vae = fallback
        if lora_string and lora_string != "None":
            try:
                model, clip = lora_fn(base_model, base_clip, lora_string)
            except Exception as e:
                print(f"[Florence2HiResFix] Failed to apply lora '{lora_string}' on session base: {e}; using base only")
                model, clip = base_model, base_clip
        else:
            model, clip = base_model, base_clip
        result = (model, clip, base_vae)
        cache[key] = result
        return result

    # Different model — full load. Validate the file exists first; without this
    # guard a missing checkpoint hits comfy/utils.load_torch_file(None,...) and
    # crashes with "'NoneType' has no attribute 'lower'". Common scenario: manifest
    # was generated on a different machine that had the per-item checkpoint
    # installed.
    try:
        import folder_paths as _fp_check
        ckpt_path_check = _fp_check.get_full_path("checkpoints", model_name)
    except Exception:
        ckpt_path_check = None
    if ckpt_path_check is None:
        print(
            f"[Florence2HiResFix] Item references model '{model_name}' but the "
            f"file is not installed in ComfyUI/models/checkpoints/. Falling back to "
            f"session default."
        )
        cache[key] = fallback
        return fallback

    try:
        model, clip, vae = ckpt_fn(
            target_model_name=model_name,
            ckpt_name=model_name,
            use_remote_vae=False,
        )
    except Exception as e:
        print(f"[Florence2HiResFix] Failed to load model '{model_name}': {e}; using session default")
        cache[key] = fallback
        return fallback

    if lora_string and lora_string != "None":
        try:
            model, clip = lora_fn(model, clip, lora_string)
        except Exception as e:
            print(f"[Florence2HiResFix] Failed to apply lora '{lora_string}': {e}; using checkpoint only")

    result = (model, clip, vae)
    cache[key] = result
    return result


def run_florence2_step(
    *,
    source_image,
    item,
    step_config,
    fallback_handles,
    ckpt_cache,
    conditioning_cache,
    positive_prompt,
    negative_prompt,
    clip_skip,
    session_model_name=None,
    model_wrap=None,
):
    """Run one Florence2 Hi-Res Fix step on one image.

    model_wrap: optional callable(model) -> model applied to the diffusion model right before
        sampling (the dashboard upscale uses it to make the sampler cancellable).

    Args:
        source_image: torch (1, H, W, 3) float32 — the PIL-converted source image
        item: manifest entry dict for this image
        step_config: dict — the upscale-step config with florence2_* fields
        fallback_handles: tuple (model, clip, vae) — already-loaded session-default handles
        ckpt_cache: dict — job-local cache for per-item ckpt+lora combos
        conditioning_cache: dict {"positive": {}, "negative": {}} — job-local conditioning cache
        positive_prompt: str — positive conditioning text
        negative_prompt: str — negative conditioning text
        clip_skip: int — clip_skip value

    Returns:
        Dict with at minimum:
          - "status": "ok" or "no_detection"
          - manifest extras: florence2_model, florence2_text_input, florence2_*
          - if status == "ok": "image_pil" (PIL), "duration" (float), "bbox", "detection_count"
    """
    import time

    t_start = time.time()
    preflight_florence2()

    # Common manifest extras regardless of outcome
    manifest_extras = {
        "florence2_model": step_config.get("florence2_model", "microsoft/Florence-2-base"),
        "florence2_text_input": step_config.get("text_input", "face"),
        "florence2_target_megapixels": float(step_config.get("target_megapixels", 1.0)),
        "florence2_crop_padding": int(step_config.get("crop_padding", 64)),
        "florence2_grow_expand": int(step_config.get("grow_expand", 32)),
        "florence2_feather": "{}/{}/{}/{}".format(
            step_config.get("feather_left", 128),
            step_config.get("feather_top", 128),
            step_config.get("feather_right", 128),
            step_config.get("feather_bottom", 128),
        ),
        "florence2_output_mask_select": step_config.get("output_mask_select", ""),
        "florence2_model_source": step_config.get("model_source", "from_manifest"),
    }

    # 1. Load Florence2 model (cached)
    florence2_model = load_florence2_model(manifest_extras["florence2_model"])

    # 2. Detect — call Florence2Run via _call_node
    try:
        from .ltx_video_generation import _call_node, _unwrap
    except ImportError:
        from ltx_video_generation import _call_node, _unwrap
    classes = get_florence2_node_classes()
    f2r_cls = classes["Florence2Run"]

    text_input = manifest_extras["florence2_text_input"]
    max_new_tokens = int(step_config.get("max_new_tokens", 1024) or 1024)
    manifest_extras["florence2_max_new_tokens"] = max_new_tokens

    # Optional pre-detect resize: Florence2's vision encoder activations scale with
    # input dims. Down-scaling the image before detection cuts the encoder's peak
    # VRAM substantially while preserving detection accuracy (faces are detectable
    # at 0.5 MP or lower without quality loss because we crop+inpaint at full source
    # resolution afterward — the mask just needs to find the region). Set
    # florence2_input_mp=0 to disable the resize.
    import comfy.utils as _cu
    src_h, src_w = int(source_image.shape[1]), int(source_image.shape[2])
    src_mp = (src_w * src_h) / (1024 * 1024)
    detect_input_mp = float(step_config.get("florence2_input_mp", 0.5) or 0)
    detect_image = source_image
    detect_resized = False
    if detect_input_mp > 0 and detect_input_mp < src_mp:
        det_w, det_h = compute_target_dims(src_w, src_h, detect_input_mp)
        det_nchw = source_image.movedim(-1, 1)
        det_resized_nchw = _cu.common_upscale(det_nchw, det_w, det_h, "lanczos", "disabled")
        detect_image = det_resized_nchw.movedim(1, -1)
        detect_resized = True
        print(
            f"[Florence2HiResFix] Detecting '{text_input}' "
            f"(src={src_w}x{src_h}={src_mp:.2f}MP -> detect={det_w}x{det_h}={detect_input_mp:.2f}MP, "
            f"max_new_tokens={max_new_tokens})"
        )
    else:
        print(
            f"[Florence2HiResFix] Detecting '{text_input}' "
            f"(src={src_w}x{src_h}={src_mp:.2f}MP, no pre-resize, max_new_tokens={max_new_tokens})"
        )
    manifest_extras["florence2_input_mp"] = detect_input_mp
    manifest_extras["florence2_detect_resized"] = detect_resized

    # max_new_tokens=1024 default (kijai stock). 256 caused polygon truncation —
    # Florence2 won't EOS cleanly under a tight cap and burns the budget on
    # garbage continuations. With detect-side resize taking the heat instead,
    # the KV cache at 1024 fits comfortably.

    # VRAM hygiene before Florence2 detect: evict SDXL/CLIP/VAE from VRAM so the
    # vision encoder + beam-search KV cache get full headroom. The standalone
    # kijai Florence2 workflow uses ~65% of 16GB at 4K/1024 tokens — but only
    # because nothing else is resident. In USCG's batch flow, SDXL (~7GB) sits
    # alongside it, leaving only ~9GB for Florence2 — under its 10GB peak on
    # runaway-token images → OOM. ComfyUI auto-reloads SDXL on the next
    # generate_image / vae.encode call below (~1-2s overhead per image).
    _vram_log("before unload + Florence2 detect")
    try:
        import comfy.model_management as _mm_pre_detect
        _mm_pre_detect.unload_all_models()
        _mm_pre_detect.soft_empty_cache()
        try:
            import torch as _torch_pre_detect
            if _torch_pre_detect.cuda.is_available():
                _torch_pre_detect.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
    _vram_log("after unload, entering Florence2 detect")

    # Settings: do_sample=True + num_beams=12 mirror kijai's reference workflow.
    # Determinism: torch.manual_seed(seed=1) is pinned inside kijai's encode().
    # NOTE: torch.inference_mode() is applied inside _call_node — see its docstring.
    det_result = _call_node(
        f2r_cls,
        image=detect_image,
        florence2_model=florence2_model,
        text_input=text_input,
        task="referring_expression_segmentation",
        fill_mask=True,
        keep_model_loaded=False,
        max_new_tokens=max_new_tokens,
        num_beams=12,
        do_sample=True,
        output_mask_select=step_config.get("output_mask_select", ""),
        seed=1,
    )
    mask = _unwrap(det_result, 1)
    _vram_log("after Florence2 detect")

    # If we ran Florence2 on a down-scaled image, the returned mask is at the
    # smaller dims. Upscale it back to the original source dims so the crop /
    # grow / feather pipeline operates at full resolution. Bilinear is fine for
    # mask resize — feather + grow downstream will smooth any edge quantization.
    if detect_resized and mask is not None and mask.numel() > 0:
        m_nchw = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        m_up_nchw = _cu.common_upscale(m_nchw, src_w, src_h, "bilinear", "disabled")
        mask = m_up_nchw.squeeze(1)

    # 3. No-detection check
    mask_sum = float(mask.sum().item()) if mask is not None else 0.0
    if mask is None or mask_sum <= 0:
        print(f"[Florence2HiResFix] No detection for '{text_input}', skipping")
        manifest_extras["status"] = "no_detection"
        manifest_extras["florence2_detection_count"] = 0
        manifest_extras["duration"] = round(time.time() - t_start, 2)
        return manifest_extras

    # 4. Mask-select parse: for referring_expression_segmentation Florence2 returns
    #    one segment; detection_count is 1 if mask is non-empty. Any OOR request -> no_detection.
    detection_count = 1
    select_indices, select_mode = parse_mask_select_indices(
        step_config.get("output_mask_select", ""), detected_count=detection_count
    )
    if select_mode == "no_detection":
        print(f"[Florence2HiResFix] mask-select '{step_config.get('output_mask_select')}' "
              f"out of range for {detection_count} detection(s), skipping")
        manifest_extras["status"] = "no_detection"
        manifest_extras["florence2_detection_count"] = detection_count
        manifest_extras["duration"] = round(time.time() - t_start, 2)
        return manifest_extras

    # 5. GrowMask (core ComfyUI)
    import nodes
    grow_cls = nodes.NODE_CLASS_MAPPINGS.get("GrowMask")
    if grow_cls is None:
        raise RuntimeError("GrowMask node not available (core ComfyUI). Reinstall ComfyUI.")
    grow_result = _call_node(
        grow_cls,
        mask=mask,
        expand=int(step_config.get("grow_expand", 32)),
        tapered_corners=True,
    )
    grown_mask = _unwrap(grow_result, 0)

    # 6. Crop by mask (pure Python)
    try:
        # min_crop_resolution / max_crop_resolution defaults are unconstrained
        # (0 / 99999) — Florence2's polygon + grow_expand + crop_padding fully
        # determine the crop region. The UI no longer exposes these fields; they
        # remain in the function signature for backwards compat with existing tests.
        cropped_img, cropped_mask, bbox = _crop_image_by_mask(
            source_image,
            grown_mask,
            padding=int(step_config.get("crop_padding", 64)),
            min_crop_resolution=int(step_config.get("min_crop_resolution", 0)),
            max_crop_resolution=int(step_config.get("max_crop_resolution", 99999)),
        )
    except ValueError as e:
        print(f"[Florence2HiResFix] Crop failed ({e}); skipping")
        manifest_extras["status"] = "no_detection"
        manifest_extras["florence2_detection_count"] = detection_count
        manifest_extras["duration"] = round(time.time() - t_start, 2)
        return manifest_extras
    crop_w, crop_h = bbox[2], bbox[3]
    if crop_w <= 1 or crop_h <= 1:
        print(f"[Florence2HiResFix] Degenerate crop ({crop_w}x{crop_h}); skipping")
        manifest_extras["status"] = "no_detection"
        manifest_extras["florence2_detection_count"] = detection_count
        manifest_extras["duration"] = round(time.time() - t_start, 2)
        return manifest_extras

    # 7. Megapixel resize (NHWC -> NCHW for common_upscale -> NHWC back)
    import comfy.utils
    requested_mp = float(step_config.get("target_megapixels", 1.0))
    target_w, target_h = compute_target_dims(crop_w, crop_h, requested_mp)
    actual_mp = (target_w * target_h) / (1024 * 1024)
    print(
        f"[Florence2HiResFix] src=({source_image.shape[2]}x{source_image.shape[1]}) "
        f"crop=({crop_w}x{crop_h}) requested_MP={requested_mp:.3f} "
        f"-> internal_pass=({target_w}x{target_h})={actual_mp:.3f}MP"
    )
    cropped_nchw = cropped_img.movedim(-1, 1)
    resized_nchw = comfy.utils.common_upscale(
        cropped_nchw, target_w, target_h, "lanczos", "disabled"
    )
    resized_img = resized_nchw.movedim(1, -1)

    # 8. Resolve checkpoint+lora handles for this item
    model_handle, clip_handle, vae_handle = _get_or_load_checkpoint_lora(
        item, fallback_handles,
        model_source=step_config.get("model_source", "from_manifest"),
        cache=ckpt_cache,
        session_model_name=session_model_name,
    )

    # 9. Conditioning. When the per-item clip is the SAME identity as the session's
    #    clip (fallback_handles[1]), we can reuse the upscale_runner's conditioning
    #    cache. When from_manifest loaded a DIFFERENT model, the per-item clip can
    #    have a different architecture (e.g. SDXL 2048-dim vs SD1.5 768-dim); reusing
    #    session-cached conditioning would feed wrong-shape tensors into cross-attn
    #    and crash with "mat1 and mat2 shapes cannot be multiplied". So we maintain
    #    a per-clip-identity conditioning cache inside ckpt_cache.
    try:
        from .batch_encoding import encode_pair_cached
    except ImportError:
        from batch_encoding import encode_pair_cached

    session_clip = fallback_handles[1]
    if clip_handle is session_clip:
        # Same clip as session — reuse the upscale_runner's shared cache.
        pos_cond, neg_cond = encode_pair_cached(
            clip_handle, conditioning_cache, positive_prompt, negative_prompt, clip_skip
        )
    else:
        # Different clip (per-manifest model) — use a per-clip-id cache inside
        # ckpt_cache so multiple images with the same per-item model still share.
        per_clip = ckpt_cache.setdefault("_cond_cache", {})
        clip_id = id(clip_handle)
        bucket = per_clip.setdefault(clip_id, {"positive": {}, "negative": {}})
        pos_cond, neg_cond = encode_pair_cached(
            clip_handle, bucket, positive_prompt, negative_prompt, clip_skip
        )

    # 10. VAE encode -> generate_image (project's known-working sampling path)
    try:
        from .image_generation import generate_image, decode_latent_with_vae
    except ImportError:
        from image_generation import generate_image, decode_latent_with_vae

    encoded_latent = vae_handle.encode(resized_img[:, :, :, :3])
    noise_seed = int(item.get("seed", 0)) + 1  # tiny offset so face pass != base seed exactly

    hires_latent_dict, _ = generate_image(
        patched_model=model_wrap(model_handle) if model_wrap is not None else model_handle,
        seed=noise_seed,
        steps=int(step_config.get("hires_steps", 15)),
        cfg=float(step_config.get("cfg", 1.5)),
        sampler_name=step_config.get("sampler", "euler"),
        scheduler=step_config.get("scheduler", "simple"),
        positive_conditioning=pos_cond,
        negative_conditioning=neg_cond,
        latent_input={"samples": encoded_latent},
        denoise=float(step_config.get("hires_denoise", 0.45)),
        width=target_w,
        height=target_h,
        # Which part of the full image this crop is (COUPLE regional prompts)
        region_frame=(bbox[1] / src_h, bbox[0] / src_w,
                      (bbox[1] + bbox[3]) / src_h, (bbox[0] + bbox[2]) / src_w),
    )
    decoded_pil = decode_latent_with_vae(vae_handle, hires_latent_dict["samples"])
    _vram_log("after sample + VAE decode")

    # Convert PIL back to tensor for paste-back
    import numpy as np
    import torch
    decoded_arr = np.array(decoded_pil).astype(np.float32) / 255.0
    decoded = torch.from_numpy(decoded_arr).unsqueeze(0)  # (1, H, W, 3)

    # 11. Resize inpainted crop back to original crop size
    decoded_nchw = decoded.movedim(-1, 1)
    resized_back_nchw = comfy.utils.common_upscale(
        decoded_nchw, crop_w, crop_h, "lanczos", "center"
    )
    resized_back = resized_back_nchw.movedim(1, -1)

    # 12. FeatherMask (core ComfyUI)
    feather_cls = nodes.NODE_CLASS_MAPPINGS.get("FeatherMask")
    if feather_cls is None:
        raise RuntimeError("FeatherMask node not available (core ComfyUI). Reinstall ComfyUI.")
    feather_result = _call_node(
        feather_cls,
        mask=cropped_mask,
        left=int(step_config.get("feather_left", 128)),
        top=int(step_config.get("feather_top", 128)),
        right=int(step_config.get("feather_right", 128)),
        bottom=int(step_config.get("feather_bottom", 128)),
    )
    feathered = _unwrap(feather_result, 0)

    # 13. Paste back into source
    final_tensor = _paste_into_image(source_image, resized_back, feathered, bbox)

    # 14. NHWC tensor -> PIL for upscale_runner save path
    from PIL import Image as PILImage
    arr = (final_tensor[0].clamp(0, 1) * 255.0).cpu().numpy().astype(np.uint8)
    final_pil = PILImage.fromarray(arr, mode="RGB")

    manifest_extras["status"] = "ok"
    manifest_extras["florence2_detection_count"] = detection_count
    manifest_extras["florence2_bbox"] = list(bbox)
    manifest_extras["image_pil"] = final_pil
    manifest_extras["image_width"] = final_pil.size[0]
    manifest_extras["image_height"] = final_pil.size[1]
    manifest_extras["duration"] = round(time.time() - t_start, 2)

    # VRAM hygiene: release intermediate tensors before next item. The KSampler
    # latents, VAE-decoded crops, and Florence2 beam-search KV cache buffers can
    # fragment GPU memory across iterations. gc.collect() drops Python-side
    # references; soft_empty_cache + torch.cuda.empty_cache let ComfyUI / PyTorch
    # reclaim the freed memory. Skip in tests (mm import would pull comfy stack).
    try:
        import gc as _gc
        _gc.collect()
        import comfy.model_management as _mm
        _mm.soft_empty_cache()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
    _vram_log("end of image (after cleanup)")

    return manifest_extras


def build_florence2_manifest_entry(
    step_result, item, *, session_name, pipeline_name, upscale_id,
    upscaled_filename, current_index, hires_denoise, base_dims=None
):
    """Build the manifest entry dict for a successful Florence2 hi-res-fix.

    Returns a dict ready to be `manifest_data["items"].insert(0, entry)`.
    """
    base = {
        k: v for k, v in item.items()
        if k not in ("id", "gen_index", "file", "filename", "upscaled",
                     "width", "height", "duration",
                     "upscale_source", "upscale_pipeline", "upscale_mode",
                     "upscale_ratio", "upscale_denoise", "upscale_model", "metrics")
    }
    base.update({
        "id": upscale_id,
        "gen_index": current_index,
        "file": f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images",
        "filename": upscaled_filename,
        "width": step_result["image_width"],
        "height": step_result["image_height"],
        "duration": float(step_result.get("duration", 0)),
        "upscaled": True,
        "upscale_source": "dashboard",
        "upscale_pipeline": pipeline_name,
        "upscale_mode": "florence2_hires",
        "florence2_model": step_result.get("florence2_model"),
        "florence2_text_input": step_result.get("florence2_text_input"),
        "florence2_target_megapixels": step_result.get("florence2_target_megapixels"),
        "florence2_crop_padding": step_result.get("florence2_crop_padding"),
        "florence2_grow_expand": step_result.get("florence2_grow_expand"),
        "florence2_feather": step_result.get("florence2_feather"),
        "florence2_output_mask_select": step_result.get("florence2_output_mask_select"),
        "florence2_model_source": step_result.get("florence2_model_source"),
        "florence2_detection_count": step_result.get("florence2_detection_count"),
        "florence2_bbox": step_result.get("florence2_bbox"),
        "hires_denoise": hires_denoise,
    })
    if base_dims:
        base.update(base_dims)
    return base


def build_florence2_no_detection_entry(step_result, item, *, sentinel_id, current_index, base_dims=None):
    """Build a sentinel manifest entry for a Florence2 no-detection result.

    No new image file is written — the entry points back at the source filename
    so the dashboard can render a "no detection" badge alongside the original.
    """
    text = step_result.get("florence2_text_input", "")
    entry = {
        "id": sentinel_id,
        "gen_index": current_index,
        "filename": item.get("filename", ""),
        "file": item.get("file", ""),
        "upscaled": False,
        "florence2_no_detection": True,
        "florence2_text_input": text,
        "florence2_model": step_result.get("florence2_model", ""),
        "note": f"Florence2 found no '{text}' in image",
    }
    if base_dims:
        entry.update(base_dims)
    return entry
