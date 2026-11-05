"""Embed a pure-nodes ComfyUI workflow into every saved image (Roadmap 71).

The dashboard's "Revise / copy as nodes" action builds a plain ComfyUI graph
(checkpoint -> LoRAs -> two prompts -> latent -> KSampler -> VAEDecode ->
PreviewImage) from a manifest item. This module is the Python twin of that
JS builder (resources/logic_comfynodes.js, pinned by tests/js/comfynodes_parity
.test.js against tests/fixtures/nodes_workflow_golden.json) plus the byte level
writers that put it where the ComfyUI frontend looks when an image is dropped
on the canvas:

* PNG  : tEXt chunks "workflow" (UI graph) and "prompt" (API graph).
* WEBP : RIFF EXIF chunk, IFD0 ASCII tags 0x010F (Make) = "workflow:<json>" and
         0x0110 (Model) = "prompt:<json>" (same layout as ComfyUI's own writer).
* JPEG : not supported (reported as such, file untouched).

Pure Python: no torch, no piexif, no network, no process spawning. Pillow is only
used for the EXIF tag encoder and to read a WEBP canvas size.
"""

import json
import math
import os
import re
import struct
import threading
import uuid
import zlib

_DEFAULT_MODEL = "XL\\waiANIPONYXL_v140_fp8_e4m3fn_full.safetensors"

PNG_SIG = b"\x89PNG\r\n\x1a\n"
EXIF_MAKE = 0x010F     # "workflow:<json>"
EXIF_MODEL = 0x0110    # "prompt:<json>"


# --------------------------------------------------------------------------- #
# JS semantics helpers
# --------------------------------------------------------------------------- #
def _truthy(v):
    """JavaScript truthiness for JSON values."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return not (v == 0 or (isinstance(v, float) and math.isnan(v)))
    if isinstance(v, str):
        return len(v) > 0
    return True


def _or(v, default):
    """`v || default` as JavaScript evaluates it."""
    return v if _truthy(v) else default


_FLOAT_RE = re.compile(r"^[\s]*([+-]?(?:Infinity|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))")


def _parse_float_js(v):
    """parseFloat(): leading-number semantics. NaN (and +-Infinity, which JSON.stringify
    turns into null) -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        m = _FLOAT_RE.match(str(v))
        if not m:
            return None
        txt = m.group(1)
        if txt.endswith("Infinity"):
            return None
        x = float(txt)
    if math.isnan(x) or math.isinf(x):
        return None
    return int(x) if x == int(x) and abs(x) < 1e15 else x


def _split_loras(lora):
    loras = []
    if _truthy(lora) and lora != "None":
        for entry in str(lora).split(" + "):
            if len(entry.strip()) == 0:
                continue
            parts = entry.split(":")
            name = parts[0]
            sm = _parse_float_js(parts[1] if len(parts) > 1 and parts[1] != "" else 1.0)
            if len(parts) > 2 and parts[2] != "":
                sc = _parse_float_js(parts[2])
            else:
                sc = sm
            loras.append({"name": name, "strength_model": sm, "strength_clip": sc})
    return loras


# --------------------------------------------------------------------------- #
# Workflow builder: exact port of buildComfyNodesWorkflow (resources/logic_comfynodes.js)
# --------------------------------------------------------------------------- #
def _node(nid, ntype, pos, size, order, inputs, outputs, widgets):
    return {
        "id": nid, "type": ntype, "pos": pos, "size": size,
        "flags": {}, "order": order, "mode": 0,
        "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def build_nodes_workflow(item, workflow_id=None):
    """Return the ComfyUI UI-format graph for a manifest item (same shape as the JS builder)."""
    d = item or {}
    loras = _split_loras(d.get("lora"))

    nid = 1
    ckpt = nid
    nid += 1
    lora_ids = []
    for _ in loras:
        lora_ids.append(nid)
        nid += 1
    pos_id = nid
    nid += 1
    neg_id = nid
    nid += 1
    lat_id = nid
    nid += 1
    ks_id = nid
    nid += 1
    vae_id = nid
    nid += 1
    prev_id = nid
    nid += 1

    if workflow_id is None:
        workflow_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"uscg:{d.get('id')}"))

    nodes = []
    n_l = len(loras)

    nodes.append(_node(
        ckpt, "CheckpointLoaderSimple", [-200, 60], [315, 98], 0, [],
        [{"name": "MODEL", "type": "MODEL", "links": []},
         {"name": "CLIP", "type": "CLIP", "links": []},
         {"name": "VAE", "type": "VAE", "links": []}],
        [_or(d.get("model"), _DEFAULT_MODEL)]))

    for index, lora in enumerate(loras):
        nodes.append(_node(
            lora_ids[index], "LoraLoader", [170 + index * 312, 60], [270, 126], index + 1,
            [{"name": "model", "type": "MODEL", "link": None},
             {"name": "clip", "type": "CLIP", "link": None}],
            [{"name": "MODEL", "type": "MODEL", "links": []},
             {"name": "CLIP", "type": "CLIP", "links": []}],
            [str(lora["name"]).replace("/", "\\"), lora["strength_model"], lora["strength_clip"]]))

    pos_x = 910 + n_l * 312

    nodes.append(_node(
        pos_id, "CLIPTextEncode", [pos_x, 3.52], [460, 190], n_l + 1,
        [{"name": "clip", "type": "CLIP", "link": None}],
        [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
        [_or(d.get("positive"), "")]))
    nodes.append(_node(
        neg_id, "CLIPTextEncode", [pos_x, 240], [470, 200], n_l + 2,
        [{"name": "clip", "type": "CLIP", "link": None}],
        [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
        [_or(d.get("negative"), "")]))
    nodes.append(_node(
        lat_id, "EmptyLatentImage", [pos_x + 130, 510], [270, 106], 1, [],
        [{"name": "LATENT", "type": "LATENT", "links": []}],
        [_or(d.get("width"), 1080), _or(d.get("height"), 1584), 1]))

    samp_x = pos_x + 510
    nodes.append(_node(
        ks_id, "KSampler", [samp_x, 23.52], [315, 708], n_l + 3,
        [{"name": "model", "type": "MODEL", "link": None},
         {"name": "positive", "type": "CONDITIONING", "link": None},
         {"name": "negative", "type": "CONDITIONING", "link": None},
         {"name": "latent_image", "type": "LATENT", "link": None}],
        [{"name": "LATENT", "type": "LATENT", "links": []}],
        [_or(d.get("seed"), 0), "fixed", _or(d.get("steps"), 25), _or(d.get("cfg"), 7),
         _or(d.get("sampler"), "dpmpp_2m"), _or(d.get("scheduler"), "karras"),
         _or(d.get("denoise"), 1)]))
    nodes.append(_node(
        vae_id, "VAEDecode", [samp_x + 390, 33], [210, 46], n_l + 4,
        [{"name": "samples", "type": "LATENT", "link": None},
         {"name": "vae", "type": "VAE", "link": None}],
        [{"name": "IMAGE", "type": "IMAGE", "links": []}], []))
    nodes.append(_node(
        prev_id, "PreviewImage", [samp_x + 370, 173], [418, 556], n_l + 5,
        [{"name": "images", "type": "IMAGE", "link": None}], [], []))

    by_id = {n["id"]: n for n in nodes}
    links = []
    state = {"link": 1}

    def wire(src, slot, dst, dslot, ltype):
        lid = state["link"]
        links.append([lid, src, slot, dst, dslot, ltype])
        by_id[src]["outputs"][slot]["links"].append(lid)
        by_id[dst]["inputs"][dslot]["link"] = lid
        state["link"] += 1

    model_src = (ckpt, 0)
    clip_src = (ckpt, 1)
    for i in range(len(loras)):
        lid_node = lora_ids[i]
        wire(model_src[0], model_src[1], lid_node, 0, "MODEL")
        wire(clip_src[0], clip_src[1], lid_node, 1, "CLIP")
        model_src = (lid_node, 0)
        clip_src = (lid_node, 1)

    wire(model_src[0], model_src[1], ks_id, 0, "MODEL")
    wire(clip_src[0], clip_src[1], pos_id, 0, "CLIP")
    wire(clip_src[0], clip_src[1], neg_id, 0, "CLIP")
    wire(ckpt, 2, vae_id, 1, "VAE")
    wire(pos_id, 0, ks_id, 1, "CONDITIONING")
    wire(neg_id, 0, ks_id, 2, "CONDITIONING")
    wire(lat_id, 0, ks_id, 3, "LATENT")
    wire(ks_id, 0, vae_id, 0, "LATENT")
    wire(vae_id, 0, prev_id, 0, "IMAGE")

    return {
        "id": workflow_id,
        "revision": 0,
        "last_node_id": nid - 1,
        "last_link_id": state["link"],   # the JS stores the NEXT id here; kept for parity
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"workflowRendererVersion": "LG", "ds": {"scale": 0.573, "offset": [488, 377]}},
        "version": 0.4,
    }


# --------------------------------------------------------------------------- #
# UI graph -> API prompt
# --------------------------------------------------------------------------- #
_WIDGET_NAMES = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "CLIPTextEncode": ["text"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", None, "steps", "cfg", "sampler_name", "scheduler", "denoise"],  # None = control_after_generate
    "VAEDecode": [],
    "PreviewImage": [],
}
_NOTE_TYPES = ("Note", "MarkdownNote")


def workflow_to_api_prompt(wf):
    """Convert the UI graph produced by build_nodes_workflow into ComfyUI's API prompt format."""
    link_src = {}
    for link in wf.get("links", []):
        link_src[link[0]] = (link[1], link[2])
    prompt = {}
    for node in wf.get("nodes", []):
        ntype = node.get("type")
        if ntype in _NOTE_TYPES:
            continue
        inputs = {}
        names = _WIDGET_NAMES.get(ntype, [])
        values = node.get("widgets_values", []) or []
        for name, val in zip(names, values):
            if name is not None:
                inputs[name] = val
        for inp in node.get("inputs", []) or []:
            lid = inp.get("link")
            if lid is not None and lid in link_src:
                src, slot = link_src[lid]
                inputs[inp["name"]] = [str(src), slot]
        prompt[str(node["id"])] = {"class_type": ntype, "inputs": inputs}
    return prompt


def nodes_metadata_for_item(item):
    """(workflow, prompt) to embed for a manifest item, or None when the item cannot be
    expressed as the plain checkpoint graph (video, sentinels, workflow sweep, GGUF /
    diffusion-model configs)."""
    if not isinstance(item, dict):
        return None
    if item.get("media_type") == "video" or item.get("florence2_no_detection"):
        return None
    if item.get("model_type") not in (None, "checkpoint"):
        return None
    base = item
    if item.get("upscaled") and item.get("base_width") and item.get("base_height"):
        base = dict(item)
        base["width"] = item["base_width"]
        base["height"] = item["base_height"]
    wf = build_nodes_workflow(base)
    if item.get("upscaled"):
        bits = ["USCG upscale:", str(item.get("upscale_mode") or "upscale")]
        if item.get("upscale_model"):
            bits.append(str(item["upscale_model"]))
        if item.get("upscale_ratio") not in (None, ""):
            bits.append(f"x{item['upscale_ratio']}")
        if item.get("upscale_denoise") not in (None, ""):
            bits.append(f"dn {item['upscale_denoise']}")
        text = " ".join(bits)
        if item.get("upscale_pipeline"):
            text += f" (pipeline {item['upscale_pipeline']})"
        nid = wf["last_node_id"] + 1
        wf["nodes"].append(_node(nid, "Note", [-200, -160], [420, 120], 0, [], [], [text]))
        wf["nodes"][-1]["properties"] = {}
        wf["last_node_id"] = nid
    return wf, workflow_to_api_prompt(wf)


# --------------------------------------------------------------------------- #
# PNG writer / reader (byte level, lossless)
# --------------------------------------------------------------------------- #
def _png_chunks(data):
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > n:
            raise ValueError("truncated PNG chunk")
        yield ctype, data[pos + 8:pos + 8 + length], data[pos:end]
        pos = end
        if ctype == b"IEND":
            break


def _png_text_key(ctype, body):
    if ctype in (b"tEXt", b"iTXt", b"zTXt"):
        i = body.find(b"\x00")
        if i > 0:
            return body[:i].decode("latin-1")
    return None


def embed_png_text(path, chunks):
    """Insert tEXt chunks {key: text} before IEND, replacing same-key text chunks."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG")
    out = [PNG_SIG]
    done = False
    for ctype, body, raw in _png_chunks(data):
        if _png_text_key(ctype, body) in chunks:
            continue
        if ctype == b"IEND":
            for key, text in chunks.items():
                payload = key.encode("latin-1") + b"\x00" + str(text).encode("latin-1", "replace")
                crc = zlib.crc32(b"tEXt" + payload) & 0xFFFFFFFF
                out.append(struct.pack(">I", len(payload)) + b"tEXt" + payload + struct.pack(">I", crc))
            done = True
        out.append(raw)
    if not done:
        raise ValueError("PNG has no IEND")
    _atomic_write(path, b"".join(out))


def _read_png_text(data):
    res = {}
    for ctype, body, _raw in _png_chunks(data):
        if ctype == b"tEXt":
            i = body.find(b"\x00")
            if i > 0:
                res[body[:i].decode("latin-1")] = body[i + 1:].decode("latin-1")
        elif ctype == b"iTXt":
            i = body.find(b"\x00")
            if i > 0 and len(body) > i + 3 and body[i + 1] == 0:   # uncompressed only
                rest = body[i + 3:]
                j = rest.find(b"\x00")
                k = rest.find(b"\x00", j + 1) if j >= 0 else -1
                if k >= 0:
                    res[body[:i].decode("latin-1")] = rest[k + 1:].decode("utf-8", "replace")
    return res


# --------------------------------------------------------------------------- #
# WEBP writer / reader (pure RIFF)
# --------------------------------------------------------------------------- #
def _riff_chunks(data):
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WEBP")
    pos = 12
    end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
    chunks = []
    while pos + 8 <= end:
        fourcc = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if len(body) < size:
            raise ValueError("truncated WEBP chunk")
        chunks.append((fourcc, body))
        pos += 8 + size + (size & 1)
    return chunks


def _riff_pack(fourcc, body):
    return fourcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")


def _exif_tiff_bytes(entries):
    from PIL import Image
    exif = Image.Exif()
    for tag, text in entries.items():
        exif[tag] = text
    raw = exif.tobytes()
    if raw.startswith(b"Exif\x00\x00"):
        raw = raw[6:]
    return raw


def embed_webp_exif(path, entries):
    """Write {tag: ascii_str} into the WEBP's EXIF chunk (IFD0), adding a VP8X header when the
    file is a simple VP8 / VP8L file. The compressed bitstream is not touched."""
    with open(path, "rb") as f:
        data = f.read()
    chunks = _riff_chunks(data)
    if not chunks:
        raise ValueError("empty WEBP")
    tiff = _exif_tiff_bytes(entries)
    names = [c[0] for c in chunks]
    if names[0] == b"VP8X":
        flags = bytearray(chunks[0][1])
        flags[0] |= 0x08
        head = [(b"VP8X", bytes(flags))]
        rest = [c for c in chunks[1:] if c[0] != b"EXIF"]
    else:
        if names[0] not in (b"VP8 ", b"VP8L"):
            raise ValueError("unsupported WEBP layout")
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        flag = 0x08
        if names[0] == b"VP8L" or b"ALPH" in names:
            flag |= 0x10
        body = bytes([flag, 0, 0, 0]) + (w - 1).to_bytes(3, "little") + (h - 1).to_bytes(3, "little")
        head = [(b"VP8X", body)]
        rest = [c for c in chunks if c[0] != b"EXIF"]
    rest.append((b"EXIF", tiff))
    payload = b"WEBP" + b"".join(_riff_pack(fc, b) for fc, b in head + rest)
    _atomic_write(path, b"RIFF" + struct.pack("<I", len(payload)) + payload)


def _read_tiff_ascii(tiff):
    """IFD0 type-2 (ASCII) entries of a TIFF blob -> {tag: str}."""
    if tiff[:6] == b"Exif\x00\x00":
        tiff = tiff[6:]
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return {}
    e = "<" if tiff[:2] == b"II" else ">"
    ifd = struct.unpack(e + "I", tiff[4:8])[0]
    if ifd + 2 > len(tiff):
        return {}
    count = struct.unpack(e + "H", tiff[ifd:ifd + 2])[0]
    out = {}
    for i in range(count):
        off = ifd + 2 + i * 12
        if off + 12 > len(tiff):
            break
        tag, typ, cnt = struct.unpack(e + "HHI", tiff[off:off + 8])
        if typ != 2:
            continue
        if cnt <= 4:
            raw = tiff[off + 8:off + 8 + cnt]
        else:
            vo = struct.unpack(e + "I", tiff[off + 8:off + 12])[0]
            raw = tiff[vo:vo + cnt]
        out[tag] = raw.split(b"\x00")[0].decode("utf-8", "replace")
    return out


def _read_webp_text(data):
    res = {}
    for fourcc, body in _riff_chunks(data):
        if fourcc == b"EXIF":
            for _tag, txt in _read_tiff_ascii(body).items():
                if ":" in txt:
                    k, v = txt.split(":", 1)
                    res[k] = v
    return res


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def _atomic_write(path, blob):
    tmp = path + ".uscg-tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)


def _sniff(path):
    with open(path, "rb") as f:
        head = f.read(12)
    if head.startswith(PNG_SIG):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def embed_comfy_workflow(path, workflow, prompt):
    """Embed workflow + prompt into a PNG or WEBP. Returns False (file untouched) for any
    other format, notably JPEG."""
    kind = _sniff(path)
    wf_txt = json.dumps(workflow, ensure_ascii=True, separators=(",", ":"))
    pr_txt = json.dumps(prompt, ensure_ascii=True, separators=(",", ":"))
    if kind == "png":
        embed_png_text(path, {"workflow": wf_txt, "prompt": pr_txt})
        return True
    if kind == "webp":
        embed_webp_exif(path, {EXIF_MAKE: "workflow:" + wf_txt, EXIF_MODEL: "prompt:" + pr_txt})
        return True
    return False


def read_comfy_metadata(path):
    """Python port of what the frontend reads on drop: {"workflow": dict|None, "prompt": dict|None}."""
    kind = _sniff(path)
    with open(path, "rb") as f:
        data = f.read()
    if kind == "png":
        txt = _read_png_text(data)
    elif kind == "webp":
        txt = _read_webp_text(data)
    else:
        txt = {}
    out = {}
    for key in ("workflow", "prompt"):
        raw = txt.get(key)
        if raw is None:
            raw = txt.get(key.capitalize())
        try:
            out[key] = json.loads(raw) if raw else None
        except ValueError:
            out[key] = None
    return out


# --------------------------------------------------------------------------- #
# Run context (mirrors image_metrics.begin_run / end_run)
# --------------------------------------------------------------------------- #
_ctx_lock = threading.Lock()
_stack = []
_warned_global = set()


class _Run:
    def __init__(self, on):
        self.on = bool(on)
        self.warned = False
        self.closed = False


def begin_run(enabled):
    """Open a run context; returns the token to hand to end_run()."""
    run = _Run(enabled)
    with _ctx_lock:
        _stack.append(run)
    return run


def end_run(token=None):
    with _ctx_lock:
        run = token if token is not None else (_stack[-1] if _stack else None)
        if run is None or run.closed:
            return
        run.closed = True
        if run in _stack:
            _stack.remove(run)


def enabled():
    """Whether the newest open run embeds workflows."""
    with _ctx_lock:
        return bool(_stack[-1].on) if _stack else False


def _warn_once(msg):
    with _ctx_lock:
        run = _stack[-1] if _stack else None
        if run is not None:
            if run.warned:
                return
            run.warned = True
        else:
            if msg in _warned_global:
                return
            _warned_global.add(msg)
    print(f"[GridTester] ⚠️ {msg}")


def stamp_file(path, item, enabled=None):
    """Embed the item's nodes workflow into the image at `path`. `enabled=None` follows the
    active run; the dashboard upscale thread passes an explicit value. Never raises.
    Returns True when a workflow was written."""
    try:
        on = globals()["enabled"]() if enabled is None else bool(enabled)
        if not on:
            return False
        meta = nodes_metadata_for_item(item)
        if meta is None:
            return False
        return bool(embed_comfy_workflow(path, meta[0], meta[1]))
    except Exception as e:  # never break a generation over metadata
        _warn_once(f"Could not embed the nodes workflow into {os.path.basename(str(path))}: {e}")
        return False


_DROP_CONF_KEYS = ("gguf_options", "lora_triggerwords_append_settings", "lora_omit_triggers", "seed_behavior",
                   "model_prompt_prefix", "model_prompt_suffix")


def upscaled_item(conf, width, height, seed, positive, negative, base_w=None, base_h=None,
                  mode=None, model=None, ratio=None, denoise=None, pipeline=None):
    """The manifest-shaped dict an upscaled save site hands to stamp_file() (the real manifest
    entry is only built after the file is written)."""
    item = {k: v for k, v in (conf or {}).items() if k not in _DROP_CONF_KEYS}
    item.update({"width": width, "height": height, "seed": seed, "positive": positive,
                 "negative": negative, "upscaled": True, "base_width": base_w, "base_height": base_h,
                 "upscale_mode": mode, "upscale_model": model, "upscale_ratio": ratio,
                 "upscale_denoise": denoise, "upscale_pipeline": pipeline})
    return item
