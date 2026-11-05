"""
Workflow Sweep: sweep any ComfyUI API-format workflow through ComfyUI's own executor.

Pure helpers (no torch / server at import time) for:
  - parsing an API-format workflow and describing it for the Builder UI,
  - suggesting sweep bindings (which literal inputs to turn into axes),
  - expanding bindings into concrete jobs (Cartesian product, cache-friendly order),
  - substituting values into the template and appending the collector node,
  - fingerprinting jobs (resume key) and building manifest items.

The runner (`workflow_sweep_runner.py`) feeds the resulting prompts to ComfyUI's queue and the
`USCGSweepCollect` node (`workflow_sweep_collect.py`) turns each job's image into a manifest item.

IMPORTANT: no network code, no subprocess, no eval/exec here (registry yara scan, brief rule 2).
Workflow JSON is only ever request-body data; nothing in this module reads a user-supplied path.
"""

import copy
import hashlib
import itertools
import json
import os
import re
import time
import random


COLLECTOR_CLASS = "USCGSweepCollect"
COLLECTOR_VIDEO_CLASS = "USCGSweepCollectVideo"
VHS_COMBINE_CLASS = "VHS_VideoCombine"
FRAME_SINKS = {  # class -> how the CreateVideo splice reads it (table order = find_video_sources order)
    "VHS_VideoCombine": {"fps": "frame_rate", "audio": "audio", "default_fps": 8.0},
    "SaveWEBM":         {"fps": "fps", "audio": None, "default_fps": 24.0},
    "SaveAnimatedWEBP": {"fps": "fps", "audio": None, "default_fps": 6.0},
    "SaveAnimatedPNG":  {"fps": "fps", "audio": None, "default_fps": 6.0},
}
CREATE_VIDEO_FPS = (1.0, 120.0)  # CreateVideo.fps MIN/MAX
MAX_JOBS = 50000

# input name -> manifest field (class-agnostic; see plan section 4.4)
ROLE_HINTS = {
    "ckpt_name": "model",
    "unet_name": "model",
    "seed": "seed",
    "noise_seed": "seed",
    "steps": "steps",
    "cfg": "cfg",
    "sampler_name": "sampler",
    "scheduler": "scheduler",
    "denoise": "denoise",
    "vae_name": "vae",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "clip_name4": "text_encoders",
}
# input name -> folder_paths type used for `folder/` syntax inside a literal axis
FOLDER_TYPES = {
    "ckpt_name": ["checkpoints"],
    "unet_name": ["diffusion_models", "unet"],
    "vae_name": ["vae"],
    "clip_name": ["text_encoders", "clip"],
    "clip_name1": ["text_encoders", "clip"],
    "clip_name2": ["text_encoders", "clip"],
    "clip_name3": ["text_encoders", "clip"],
    "clip_name4": ["text_encoders", "clip"],
    "lora_name": ["loras"],
}
# manifest fields a literal axis may feed
AXIS_FIELDS = ("model", "sampler", "scheduler", "steps", "cfg", "seed", "denoise", "vae")
# primitive-string-ish input names a prompt text can hide behind
_TEXT_INPUTS = ("value", "string", "text", "prompt", "multiline_text")
# text key in a "text link" hop chain, tried in this order when following links
_FOLLOW_INPUTS = ("source", "on_false", "on_true", "text", "value", "string", "prompt")
_RES_RE = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")


# --------------------------------------------------------------------------- parsing

def is_ui_format(obj):
    """True for the editor (UI) workflow format, which has `nodes` + `links` lists."""
    return isinstance(obj, dict) and isinstance(obj.get("nodes"), list) and isinstance(obj.get("links"), list)


def is_link(value):
    """API-format link: [node_id, output_index]."""
    return (
        isinstance(value, (list, tuple)) and len(value) == 2
        and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
        and isinstance(value[1], int) and not isinstance(value[1], bool)
    )


def parse_api_workflow(data):
    """
    Parse an API-format workflow (ComfyUI "Export (API)"), from a JSON string or an object.

    Returns the prompt dict {node_id(str): {class_type, inputs, _meta?}} unchanged.
    Raises ValueError with a readable message otherwise.
    """
    if isinstance(data, (str, bytes)):
        text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        if not text.strip():
            raise ValueError("The workflow JSON is empty.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"The workflow is not valid JSON: {e}")
    if is_ui_format(data):
        raise ValueError(
            "This is an editor-format workflow (it has 'nodes' and 'links'). "
            "In ComfyUI use Workflow > Export (API) and load that file instead."
        )
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict) and not any(
            isinstance(v, dict) and "class_type" in v for v in data.values()):
        data = data["prompt"]  # a saved /prompt request body
    if not isinstance(data, dict) or not data:
        raise ValueError("An API-format workflow is a JSON object of numbered nodes.")
    prompt = {}
    for nid, node in data.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ValueError(f"Node '{nid}' has no class_type; this does not look like an API-format workflow.")
        if not isinstance(node.get("inputs", {}), dict):
            raise ValueError(f"Node '{nid}' has malformed inputs.")
        node.setdefault("inputs", {})
        prompt[str(nid)] = node
    return prompt


def node_title(prompt, node_id):
    node = prompt.get(str(node_id)) or {}
    return (node.get("_meta") or {}).get("title") or node.get("class_type") or str(node_id)


def _node_sort_key(nid):
    s = str(nid)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def comfy_class_info(class_type):
    """
    Describe a node class from ComfyUI's registry (V1 and V3 alike), or None when ComfyUI is not
    importable (tests). Returns {missing, return_types, output_node, inputs:{name:{type,options,required}}}.
    """
    try:
        import nodes  # ComfyUI
        cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    except Exception:
        return None
    if cls is None:
        return {"missing": True, "return_types": [], "output_node": False, "inputs": {}}
    inputs = {}
    try:
        spec_all = cls.INPUT_TYPES() or {}
    except Exception:
        spec_all = {}
    for section in ("required", "optional"):
        for name, spec in (spec_all.get(section) or {}).items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            options = None
            if isinstance(t, (list, tuple)):
                options, t = list(t), "COMBO"
            elif t == "COMBO" and isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
                options = list(spec[1].get("options") or [])
            inputs[name] = {"type": t if isinstance(t, str) else str(t), "options": options,
                            "required": section == "required"}
    try:
        return_types = [str(x) for x in (getattr(cls, "RETURN_TYPES", ()) or ())]
    except Exception:
        return_types = []
    try:
        output_node = bool(getattr(cls, "OUTPUT_NODE", False))
    except Exception:
        output_node = False
    return {"missing": False, "return_types": return_types, "output_node": output_node, "inputs": inputs}


def _info(class_info, class_type):
    if class_info is None:
        return None
    try:
        return class_info(class_type)
    except Exception:
        return None


# --------------------------------------------------------------------------- graph helpers

def ancestors_ordered(prompt, node_id):
    """Breadth-first list of node ids upstream of node_id, starting with node_id itself.
    Closest nodes come first, so 'first ancestor wins' picks the node nearest the output."""
    start = str(node_id)
    if start not in prompt:
        return []
    seen, order, queue = {start}, [], [start]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for value in (prompt[cur].get("inputs") or {}).values():
            if is_link(value):
                up = str(value[0])
                if up in prompt and up not in seen:
                    seen.add(up)
                    queue.append(up)
    return order


def graph_descendants(prompt, restrict_to=None):
    """{node_id: number of nodes that (transitively) depend on it}, optionally restricted to a node set."""
    nodes_ = set(restrict_to) if restrict_to is not None else set(prompt)
    downstream = {n: set() for n in nodes_}
    for nid in nodes_:
        for value in (prompt[nid].get("inputs") or {}).values():
            if is_link(value) and str(value[0]) in nodes_:
                downstream[str(value[0])].add(nid)
    memo = {}

    def reach(n, stack=()):
        if n in memo:
            return memo[n]
        out = set()
        for d in downstream.get(n, ()):
            if d in stack:
                continue
            out.add(d)
            out |= reach(d, stack + (n,))
        memo[n] = out
        return out

    return {n: len(reach(n)) for n in nodes_}


def find_image_sources(prompt, class_info=None):
    """
    Every IMAGE socket a collector could be spliced onto, best candidate first:
    the feeder of the first SaveImage, then PreviewImage feeders, then any node whose
    class advertises an IMAGE output (needs ComfyUI's registry via class_info).
    Returns [{node_id, output_index, type, picked_from?}].
    """
    sources, seen = [], set()

    def add(nid, idx, picked_from=None):
        key = (str(nid), int(idx))
        if key in seen or str(nid) not in prompt:
            return
        seen.add(key)
        entry = {"node_id": key[0], "output_index": key[1], "type": "IMAGE"}
        if picked_from is not None:
            entry["picked_from"] = str(picked_from)
        sources.append(entry)

    def feeders(class_pred):
        for nid in sorted(prompt, key=_node_sort_key):
            ct = prompt[nid].get("class_type", "")
            if not class_pred(ct):
                continue
            for iname, value in (prompt[nid].get("inputs") or {}).items():
                if iname in ("images", "image") and is_link(value):
                    add(value[0], value[1], picked_from=nid)

    feeders(lambda ct: ct == "SaveImage" or ct.startswith("SaveImage"))
    feeders(lambda ct: ct == "PreviewImage" or ct.startswith("PreviewImage"))
    for nid in sorted(prompt, key=_node_sort_key):
        info = _info(class_info, prompt[nid].get("class_type"))
        if info and not info.get("missing"):
            for idx, rt in enumerate(info.get("return_types") or []):
                if rt == "IMAGE":
                    add(nid, idx)
    return sources


def find_output_nodes(prompt, class_info=None):
    """Output nodes of the workflow (SaveImage-like or OUTPUT_NODE classes) with their image source."""
    outs = []
    for nid in sorted(prompt, key=_node_sort_key):
        ct = prompt[nid].get("class_type", "")
        info = _info(class_info, ct)
        is_out = bool(info and info.get("output_node")) or ct.startswith(
            ("SaveImage", "PreviewImage", "PreviewAny", "SaveVideo", "SaveWEBM", "SaveAnimated", "VHS_VideoCombine"))
        if not is_out:
            continue
        src = None
        for iname in ("images", "image", "source", "video"):
            v = (prompt[nid].get("inputs") or {}).get(iname)
            if is_link(v):
                src = [str(v[0]), int(v[1])]
                break
        outs.append({"node_id": nid, "class_type": ct, "title": node_title(prompt, nid), "source": src})
    return outs


def _frames_unsupported(node):
    """Why a frame-sink node cannot be a frames source, or None when it can (only VHS has restrictions)."""
    if (node or {}).get("class_type") != VHS_COMBINE_CLASS:
        return None
    inputs = (node or {}).get("inputs") or {}
    if is_link(inputs.get("vae")):
        return "it has a `vae` link (latent input): decode the latents before VHS Video Combine"
    if is_link(inputs.get("meta_batch")):
        return "it has a `meta_batch` link (batch manager is not supported)"
    return None


_vhs_unsupported = _frames_unsupported


def _sink_fps_raw(node):
    """The raw fps input of a frame-sink node (a link, a number or the sink's default)."""
    sink = FRAME_SINKS.get((node or {}).get("class_type"), FRAME_SINKS[VHS_COMBINE_CLASS])
    fps = ((node or {}).get("inputs") or {}).get(sink["fps"], sink["default_fps"])
    if is_link(fps):
        return [str(fps[0]), int(fps[1])]
    try:
        return float(fps)
    except (TypeError, ValueError):
        return float(sink["default_fps"])


def _sink_fps(node):
    """(value_or_link, clamped) for the CreateVideo fps: a link is forwarded unchanged, a literal is
    clamped to CreateVideo's 1-120."""
    fps = _sink_fps_raw(node)
    if isinstance(fps, list):
        return fps, False
    lo, hi = CREATE_VIDEO_FPS
    clamped = min(max(fps, lo), hi)
    return clamped, clamped != fps


def find_video_sources(prompt, class_info=None, warnings=None):
    """
    Every socket a video collector could be fed from, best candidate first:
      1. feeders of SaveVideo*.video            (kind "video", VIDEO socket)
      2. feeders of VHS_VideoCombine / SaveWEBM / SaveAnimatedWEBP / SaveAnimatedPNG .images (kind "frames", IMAGE socket, converted with a CreateVideo)
      3. any node whose class advertises a VIDEO output (CreateVideo, LoadVideo, ...)
    Returns [{node_id, output_index, type, media:"video", kind, frames_from?, picked_from?}].
    Unsupported VHS nodes are skipped with an entry in `warnings` (when a list is given).
    """
    sources, seen = [], set()

    def add(nid, idx, rtype, kind, picked_from=None, frames_from=None):
        key = (str(nid), int(idx), kind, frames_from)
        if key in seen or str(nid) not in prompt:
            return
        seen.add(key)
        entry = {"node_id": key[0], "output_index": key[1], "type": rtype, "media": "video", "kind": kind}
        if frames_from is not None:
            entry["frames_from"] = str(frames_from)
        if picked_from is not None:
            entry["picked_from"] = str(picked_from)
        sources.append(entry)

    for nid in sorted(prompt, key=_node_sort_key):
        ct = prompt[nid].get("class_type", "")
        v = (prompt[nid].get("inputs") or {}).get("video")
        if ct.startswith("SaveVideo") and is_link(v):
            add(v[0], v[1], "VIDEO", "video", picked_from=nid)
    for sink_class in FRAME_SINKS:
        for nid in sorted(prompt, key=_node_sort_key):
            if prompt[nid].get("class_type") != sink_class:
                continue
            v = (prompt[nid].get("inputs") or {}).get("images")
            if not is_link(v):
                continue
            why = _frames_unsupported(prompt[nid])
            if why:
                if warnings is not None:
                    warnings.append(f"{node_title(prompt, nid)} (#{nid}) was skipped as a video source: {why}.")
                continue
            add(v[0], v[1], "IMAGE", "frames", picked_from=nid, frames_from=nid)
    for nid in sorted(prompt, key=_node_sort_key):
        ct = prompt[nid].get("class_type")
        info = _info(class_info, ct)
        if info is None:
            if ct == "CreateVideo":  # no registry (tests): the one core node we know produces VIDEO
                add(nid, 0, "VIDEO", "video")
            continue
        if not info.get("missing"):
            for idx, rt in enumerate(info.get("return_types") or []):
                if rt == "VIDEO":
                    add(nid, idx, "VIDEO", "video")
    return sources


def source_to_output(src):
    """A source entry -> the stored `workflow_output` shape. Image outputs stay {node_id, output_index} only."""
    out = {"node_id": str(src["node_id"]), "output_index": int(src.get("output_index", 0) or 0)}
    if src.get("media") == "video":
        out["media"] = "video"
        if src.get("frames_from") is not None:
            out["frames_from"] = str(src["frames_from"])
    return out


def default_output(prompt, class_info=None):
    """
    The socket that should feed the collector, in `workflow_output` shape, or None.
    A SaveImage / PreviewImage feeder wins (image workflows behave as they always did); otherwise the
    best video source; otherwise any IMAGE socket.
    """
    images = find_image_sources(prompt, class_info)
    if images and images[0].get("picked_from") is not None:
        return images[0]
    videos = find_video_sources(prompt, class_info)
    if videos:
        return source_to_output(videos[0])
    return images[0] if images else None


def describe_nodes(prompt, class_info=None):
    """Node table for the Builder: one entry per node with typed inputs (links flagged)."""
    out = []
    for nid in sorted(prompt, key=_node_sort_key):
        node = prompt[nid]
        ct = node.get("class_type", "")
        info = _info(class_info, ct)
        specs = (info or {}).get("inputs") or {}
        inputs = []
        for iname, value in (node.get("inputs") or {}).items():
            spec = specs.get(iname) or {}
            row = {"name": iname, "value": None if is_link(value) else value, "is_link": is_link(value),
                   "link": [str(value[0]), int(value[1])] if is_link(value) else None,
                   "type": spec.get("type") or _guess_type(value), "options": spec.get("options")}
            inputs.append(row)
        out.append({"id": nid, "class_type": ct, "title": node_title(prompt, nid),
                    "missing_class": bool(info and info.get("missing")), "inputs": inputs})
    return out


def _guess_type(value):
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    return "UNKNOWN"


# --------------------------------------------------------------------------- suggestions

def _follow_text_source(prompt, node_id, input_name, max_hops=6):
    """CLIPTextEncode.text is often a LINK (MOSS: 30.text <- 34 <- 35 <- 26.value). Find the first
    node upstream that holds a literal string; returns {node_id, input} or None."""
    node = prompt.get(str(node_id))
    if not node:
        return None
    v = (node.get("inputs") or {}).get(input_name)
    if isinstance(v, str):
        return {"node_id": str(node_id), "input": input_name}
    if not is_link(v):
        return None
    seen = {str(node_id)}
    queue = [(str(v[0]), 1)]
    while queue:
        cur, depth = queue.pop(0)
        if cur in seen or cur not in prompt or depth > max_hops:
            continue
        seen.add(cur)
        inputs = prompt[cur].get("inputs") or {}
        for key in _TEXT_INPUTS:
            if isinstance(inputs.get(key), str):
                return {"node_id": cur, "input": key}
        for key in _FOLLOW_INPUTS:
            link = inputs.get(key)
            if is_link(link):
                queue.append((str(link[0]), depth + 1))
        for key, link in inputs.items():
            if key not in _FOLLOW_INPUTS and is_link(link) and key not in ("clip", "model", "vae"):
                queue.append((str(link[0]), depth + 1))
    return None


def find_prompt_targets(prompt, output_node_id):
    """Locate the positive / negative text inputs feeding the sampler(s) upstream of the output.
    Class-agnostic: any node with link inputs named `positive` and `negative` (KSampler, CFGGuider...).
    Returns {"positive": {node_id,input}|None, "negative": ...}."""
    result = {"positive": None, "negative": None}
    for nid in ancestors_ordered(prompt, output_node_id):
        inputs = prompt[nid].get("inputs") or {}
        for role in ("positive", "negative"):
            if result[role] is not None or not is_link(inputs.get(role)):
                continue
            enc = str(inputs[role][0])
            enc_inputs = (prompt.get(enc) or {}).get("inputs") or {}
            for key in ("text", "prompt", "text_g", "value"):
                if key in enc_inputs:
                    target = _follow_text_source(prompt, enc, key)
                    if target:
                        result[role] = target
                        break
    return result


def _lora_slot_nodes(prompt, anc):
    slots = []
    for nid in anc:
        inputs = prompt[nid].get("inputs") or {}
        if "lora_name" in inputs and "strength_model" in inputs:
            slots.append(nid)
    return slots


def suggest_bindings(prompt, output, class_info=None):
    """Binding suggestions for the Builder, restricted to nodes upstream of the output.
    Returns [{kind, field, targets, reason}] (targets use {node_id, input})."""
    out_id = str(output["node_id"] if isinstance(output, dict) else output[0])
    anc = ancestors_ordered(prompt, out_id)
    suggestions = []
    targets = find_prompt_targets(prompt, out_id)
    for role in ("positive", "negative"):
        if targets[role]:
            suggestions.append({"kind": role, "field": None, "targets": [targets[role]],
                                "reason": f"{role} prompt text feeding the sampler"})
    for nid in anc:  # nearest ancestor first, so the first candidate of a field is the first-ancestor-wins pick
        for iname, value in (prompt[nid].get("inputs") or {}).items():
            field = ROLE_HINTS.get(iname)
            if not field or field == "text_encoders" or is_link(value) or isinstance(value, (list, dict)):
                continue
            suggestions.append({"kind": "literal", "field": field,
                                "targets": [{"node_id": nid, "input": iname}],
                                "reason": f"{node_title(prompt, nid)}.{iname}"})
    slots = _lora_slot_nodes(prompt, anc)
    if slots:
        suggestions.append({"kind": "lora", "field": "lora", "slots": [{"node_id": s} for s in slots],
                            "targets": [], "reason": "LoRA loader node(s) upstream of the output"})
    for nid in anc:
        inputs = prompt[nid].get("inputs") or {}
        ct = prompt[nid].get("class_type", "")
        if "width" in inputs and "height" in inputs and "Latent" in ct and "Empty" in ct \
                and not is_link(inputs["width"]) and not is_link(inputs["height"]):
            suggestions.append({"kind": "resolution", "field": None, "targets": [],
                                "width": {"node_id": nid, "input": "width"},
                                "height": {"node_id": nid, "input": "height"},
                                "reason": f"{node_title(prompt, nid)} width x height"})
            break
    return suggestions


# --------------------------------------------------------------------------- template -> known fields

def _field_sources(prompt, output_id):
    """Static (field, node_id, input) triples, first ancestor wins, plus encoder/clip_type/lora sources."""
    anc = ancestors_ordered(prompt, output_id)
    sources = {}
    encoders = []
    for nid in anc:
        ct = prompt[nid].get("class_type", "")
        for iname, value in (prompt[nid].get("inputs") or {}).items():
            field = ROLE_HINTS.get(iname)
            if field == "text_encoders":
                encoders.append((nid, iname))
            elif field and field not in sources:
                sources[field] = (nid, iname)
            elif iname == "type" and "CLIPLoader" in ct and "clip_type" not in sources:
                sources["clip_type"] = (nid, iname)
    prompt_targets = find_prompt_targets(prompt, output_id)
    for role in ("positive", "negative"):
        if prompt_targets[role]:
            sources[role] = (prompt_targets[role]["node_id"], prompt_targets[role]["input"])
    return sources, encoders, _lora_slot_nodes(prompt, anc)


def _literal(prompt, nid, iname):
    v = ((prompt.get(nid) or {}).get("inputs") or {}).get(iname)
    return None if is_link(v) or isinstance(v, (dict,)) else v


def known_fields(prompt, sources, encoders, lora_nodes):
    """Manifest fields read from a (substituted) prompt using precomputed sources."""
    fields = {}
    for field, (nid, iname) in sources.items():
        v = _literal(prompt, nid, iname)
        if v is not None:
            fields[field] = v
    enc = [_literal(prompt, nid, iname) for nid, iname in encoders]
    enc = [e for e in enc if isinstance(e, str) and e]
    if enc:
        fields["text_encoders"] = enc
    parts = []
    for nid in lora_nodes:
        inputs = prompt[nid].get("inputs") or {}
        name = inputs.get("lora_name")
        ms, cs = inputs.get("strength_model"), inputs.get("strength_clip", inputs.get("strength_model"))
        if isinstance(name, str) and name and not is_link(ms) and not is_link(cs):
            try:
                if float(ms) == 0 and float(cs) == 0:
                    continue
                parts.append(f"{name}:{float(ms):g}:{float(cs):g}")
            except (TypeError, ValueError):
                parts.append(name)
    fields["lora"] = " + ".join(parts) if parts else "None"
    fields["lora_expanded"] = fields["lora"]
    if fields.get("denoise") == 1.0:
        fields["denoise"] = 1
    return fields


def known_fields_from_template(prompt, output_id):
    sources, encoders, lora_nodes = _field_sources(prompt, str(output_id))
    return known_fields(prompt, sources, encoders, lora_nodes)


# --------------------------------------------------------------------------- coercion, prompts, fingerprints

def coerce_like(value, template_value):
    """Coerce a sweep value to the type of the template literal it replaces
    (int stays int, integral float -> int for INT inputs, "true"/"false" -> bool, combos as str)."""
    if template_value is None or is_link(template_value):
        return value
    try:
        if isinstance(template_value, bool):
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "on"):
                    return True
                if low in ("false", "0", "no", "off"):
                    return False
            return bool(value)
        if isinstance(template_value, int):
            f = float(value)
            return int(f) if f == int(f) else f
        if isinstance(template_value, float):
            return float(value)
        if isinstance(template_value, str):
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def template_hash(prompt, output, run_other_outputs=False, media="image", frames_from=None):
    body = {"p": prompt, "o": [str(output[0]), int(output[1])], "r": bool(run_other_outputs)}
    if media == "video":  # image templates keep the exact payload they always had (resume of old sessions)
        body["m"], body["f"] = "video", (None if frames_from is None else str(frames_from))
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _strip_meta(prompt):
    return {nid: {k: v for k, v in node.items() if k != "_meta"} for nid, node in prompt.items()}


def fingerprint(substituted_prompt, output, media="image", frames_from=None):
    """Resume key of one job: sha256 of the substituted template (no _meta, no collector) + output socket.
    Video collections add the media kind (and the VHS node for frames), so a video and an image
    collection of the same socket never skip each other; image fingerprints are unchanged."""
    body = {"p": _strip_meta(substituted_prompt), "o": [str(output[0]), int(output[1])]}
    if media == "video":
        body["m"], body["f"] = "video", (None if frames_from is None else str(frames_from))
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def apply_subs(template, subs):
    """Deepcopy the template and set each (node_id, input, value)."""
    prompt = copy.deepcopy(template)
    for nid, iname, value in subs:
        nid = str(nid)
        if nid not in prompt:
            raise ValueError(f"Binding targets node {nid}, which is not in the workflow.")
        prompt[nid].setdefault("inputs", {})[iname] = value
    return prompt


def collector_node_id(prompt):
    """Numeric string one above the highest numeric id (keeps ComfyUI's history 'load' path happy)."""
    nums = [int(k) for k in prompt if str(k).isdigit()]
    cid = (max(nums) + 1) if nums else 1
    while str(cid) in prompt:
        cid += 1
    return str(cid)


def build_job_prompt(template, subs, output, collector_inputs, media="image", frames_from=None):
    """
    Substitute `subs` into a copy of the template and append the collector node.
    Nothing else is touched, except for a frames-sourced video (VHS), where a core `CreateVideo` node is
    spliced in front of the collector (images / fps / audio copied from the VHS node of the substituted
    prompt), so the collector only ever sees a VIDEO. Returns (prompt, collector_id).
    """
    prompt = apply_subs(template, subs)
    if media != "video":
        cid = collector_node_id(prompt)
        inputs = {"images": [str(output[0]), int(output[1])]}
        inputs.update(collector_inputs or {})
        prompt[cid] = {"class_type": COLLECTOR_CLASS, "inputs": inputs, "_meta": {"title": "USCG Sweep Collect"}}
        return prompt, cid
    source = [str(output[0]), int(output[1])]
    if frames_from is not None and str(frames_from) in prompt:
        sink_node = prompt[str(frames_from)]
        sink = FRAME_SINKS.get(sink_node.get("class_type"), FRAME_SINKS[VHS_COMBINE_CLASS])
        vhs_inputs = sink_node.get("inputs") or {}
        images = vhs_inputs.get("images")
        cv_inputs = {"images": [str(images[0]), int(images[1])] if is_link(images) else source}
        cv_inputs["fps"] = _sink_fps(sink_node)[0]
        audio = vhs_inputs.get(sink["audio"]) if sink["audio"] else None
        if is_link(audio):
            cv_inputs["audio"] = [str(audio[0]), int(audio[1])]
        cv_id = collector_node_id(prompt)
        prompt[cv_id] = {"class_type": "CreateVideo", "inputs": cv_inputs, "_meta": {"title": "USCG Sweep Create Video"}}
        source = [cv_id, 0]
    cid = collector_node_id(prompt)
    inputs = {"video": source}
    inputs.update(collector_inputs or {})
    prompt[cid] = {"class_type": COLLECTOR_VIDEO_CLASS, "inputs": inputs, "_meta": {"title": "USCG Sweep Collect Video"}}
    return prompt, cid


# --------------------------------------------------------------------------- expansion

def _cu():
    try:
        from . import config_utils
    except ImportError:
        import config_utils
    return config_utils


def _parse_prompts(raw, fallback):
    """Same rules as config_utils.expand_configs per-config prompt handling."""
    cu = _cu()
    if isinstance(raw, (list, tuple)):
        vals = cu.parse_prompt_input_nested(json.dumps(raw))
    elif isinstance(raw, str) and raw.strip():
        vals = cu.parse_prompt_input_nested(raw)
    else:
        vals = list(fallback or [])
    return [v for v in vals if isinstance(v, str)]


def _expand_folder_value(value, iname):
    if not (isinstance(value, str) and value.replace("\\", "/").endswith("/")):
        return [value]
    cu = _cu()
    for folder_type in FOLDER_TYPES.get(iname, []):
        try:
            found = cu.get_files_from_folder(value, folder_type)
        except Exception:
            found = []
        if found:
            return found
    return []


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        key = json.dumps(v, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _is_loader_class(class_info, class_type):
    info = _info(class_info, class_type)
    if info and info.get("return_types"):
        return any(t in ("MODEL", "CLIP", "VAE") for t in info["return_types"]) and not info.get("inputs", {}).get("model") \
            and not info.get("inputs", {}).get("clip")
    return "Loader" in class_type or "Checkpoint" in class_type


class _Axis:
    __slots__ = ("label", "kind", "field", "options", "nodes", "binding_id", "loader")

    def __init__(self, label, kind, field, options, nodes, binding_id, loader=False):
        self.label, self.kind, self.field = label, kind, field
        self.options = options          # list of {display, subs, fields}
        self.nodes = nodes              # node ids the axis writes to
        self.binding_id = binding_id
        self.loader = loader


def _label_for(prompt, nid, iname, title_counts):
    title = node_title(prompt, nid)
    return f"{title}.{iname} #{nid}" if title_counts.get(title, 0) > 1 else f"{title}.{iname}"


def _as_target(t):
    if not isinstance(t, dict) or "node_id" not in t or "input" not in t:
        raise ValueError(f"A binding target needs node_id and input, got {t!r}.")
    return str(t["node_id"]), str(t["input"])


def _build_axes(prompt, bindings, entry, anc_set, extra_seeds, pos_prompts, neg_prompts, class_info, warnings):
    cu = _cu()
    title_counts = {}
    for nid in prompt:
        t = node_title(prompt, nid)
        title_counts[t] = title_counts.get(t, 0) + 1
    axes = []
    written = {}  # (node, input) -> binding id, to catch two bindings writing the same input

    def claim(nid, iname, bid, override_link):
        if nid not in prompt:
            raise ValueError(f"Binding {bid}: node {nid} is not in the workflow.")
        cur = (prompt[nid].get("inputs") or {}).get(iname)
        if is_link(cur) and not override_link:
            raise ValueError(
                f"Binding {bid}: {node_title(prompt, nid)}.{iname} is linked to another node. "
                "Tick 'override link' to replace the link with the swept value.")
        if (nid, iname) in written:
            raise ValueError(f"Bindings {written[(nid, iname)]} and {bid} both write {node_title(prompt, nid)}.{iname}.")
        written[(nid, iname)] = bid
        if nid not in anc_set:
            warnings.append(f"Binding {bid} ({node_title(prompt, nid)}.{iname}) is not upstream of the chosen output; "
                            "it has no effect (the image would come from cache).")

    for bi, b in enumerate(bindings):
        if not isinstance(b, dict):
            continue
        bid = str(b.get("id") or f"b{bi + 1}")
        kind = b.get("kind", "literal")
        override = bool(b.get("override_link"))

        if kind == "literal":
            targets = [_as_target(t) for t in (b.get("targets") or [])]
            if not targets:
                raise ValueError(f"Binding {bid} has no targets.")
            for nid, iname in targets:
                claim(nid, iname, bid, override)
            nid0, in0 = targets[0]
            tmpl = (prompt[nid0].get("inputs") or {}).get(in0)
            if "field" in b:
                field = b.get("field") or None
            else:
                field = ROLE_HINTS.get(in0)
                field = field if field in AXIS_FIELDS else None
            if field is not None and field not in AXIS_FIELDS:
                raise ValueError(f"Binding {bid}: unknown manifest field '{field}'.")
            raw_values = b.get("values")
            raw_values = raw_values if isinstance(raw_values, list) else ([] if raw_values is None else [raw_values])
            values = []
            for v in raw_values:
                values.extend(_expand_folder_value(v, in0))
            values = [coerce_like(v, tmpl) for v in values]
            if field == "seed":
                values = values + [coerce_like(s, tmpl) for s in (extra_seeds or [])]
            values = _dedupe(values)
            if not values:
                raise ValueError(f"Binding {bid} ({node_title(prompt, nid0)}.{in0}) has no values.")
            options = [{"display": v, "subs": [(n, i, v) for n, i in targets],
                        "fields": ({field: v} if field else {})} for v in values]
            loader = _is_loader_class(class_info, prompt[nid0].get("class_type", ""))
            axes.append(_Axis(_label_for(prompt, nid0, in0, title_counts), "literal", field, options,
                              [n for n, _ in targets], bid, loader))

        elif kind in ("positive", "negative"):
            targets = [_as_target(t) for t in (b.get("targets") or [])]
            if not targets:
                raise ValueError(f"Binding {bid} ({kind}) has no targets.")
            for nid, iname in targets:
                claim(nid, iname, bid, override)
            fallback = pos_prompts if kind == "positive" else neg_prompts
            prompts = _dedupe(_parse_prompts(entry.get(kind, ""), fallback))
            if not prompts or all(not p.strip() for p in prompts):
                raise ValueError(
                    f"Binding {bid} swaps the {kind} prompt but there are no {kind} prompts. "
                    "Add them in this config's Prompts section (or the global prompts).")
            options = [{"display": p, "subs": [(n, i, p) for n, i in targets], "fields": {kind: p}} for p in prompts]
            axes.append(_Axis(_label_for(prompt, targets[0][0], targets[0][1], title_counts), kind, kind, options,
                              [n for n, _ in targets], bid))

        elif kind == "lora":
            slots = [str(s["node_id"] if isinstance(s, dict) else s) for s in (b.get("slots") or [])]
            if not slots:
                raise ValueError(f"Binding {bid} (LoRA) has no slots.")
            for nid in slots:
                if nid not in prompt:
                    raise ValueError(f"Binding {bid}: LoRA node {nid} is not in the workflow.")
                inputs = prompt[nid].get("inputs") or {}
                for iname in ("lora_name", "strength_model", "strength_clip"):
                    if iname in inputs:
                        claim(nid, iname, bid, override)
            raw = b.get("values")
            raw = raw if isinstance(raw, list) else ([] if raw is None else [raw])
            lora_strings = _dedupe(cu.expand_lora_stack([str(v) for v in raw])) if raw else []
            if not lora_strings:
                raise ValueError(f"Binding {bid} (LoRA) has no values.")
            options = []
            for s in lora_strings:
                parts = [] if s == "None" else cu.parse_lora_definition(s)
                if len(parts) > len(slots):
                    raise ValueError(
                        f"Binding {bid}: '{s}' stacks {len(parts)} LoRAs but only {len(slots)} LoRA slot(s) are bound.")
                subs = []
                for si, nid in enumerate(slots):
                    inputs = prompt[nid].get("inputs") or {}
                    if si < len(parts):
                        name, ms, cs = parts[si]
                        subs.append((nid, "lora_name", name))
                        subs.append((nid, "strength_model", ms))
                        if "strength_clip" in inputs:
                            subs.append((nid, "strength_clip", cs))
                    else:
                        subs.append((nid, "strength_model", 0.0))
                        if "strength_clip" in inputs:
                            subs.append((nid, "strength_clip", 0.0))
                options.append({"display": s, "subs": subs, "fields": {"lora": s, "lora_expanded": s}})
            axes.append(_Axis(_label_for(prompt, slots[0], "lora_name", title_counts), "lora", "lora", options, slots, bid, True))

        elif kind == "resolution":
            try:
                wt, ht = _as_target(b.get("width")), _as_target(b.get("height"))
            except (ValueError, TypeError):
                raise ValueError(f"Binding {bid} (resolution) needs width and height targets.")
            claim(wt[0], wt[1], bid, override)
            claim(ht[0], ht[1], bid, override)
            raw = b.get("values")
            raw = raw if isinstance(raw, list) else ([] if raw is None else [raw])
            options = []
            for v in _dedupe([str(x) for x in raw]):
                m = _RES_RE.match(v)
                if not m:
                    raise ValueError(f"Binding {bid}: '{v}' is not a WxH resolution like 1024x1024.")
                w, h = int(m.group(1)), int(m.group(2))
                options.append({"display": f"{w}x{h}", "subs": [(wt[0], wt[1], w), (ht[0], ht[1], h)],
                                "fields": {"width": w, "height": h}})
            if not options:
                raise ValueError(f"Binding {bid} (resolution) has no values.")
            axes.append(_Axis(_label_for(prompt, wt[0], wt[1], title_counts), "resolution", None, options,
                              [wt[0], ht[0]], bid))
        else:
            raise ValueError(f"Binding {bid}: unknown kind '{kind}'.")
    return axes


def _pair_prompt_axes(axes):
    """Prompt pairing rule of expand_configs: both lists > 1 and equal length -> 1-to-1, else product."""
    pos = [a for a in axes if a.kind == "positive"]
    neg = [a for a in axes if a.kind == "negative"]
    if len(pos) == 1 and len(neg) == 1 and len(pos[0].options) > 1 and len(pos[0].options) == len(neg[0].options):
        p, n = pos[0], neg[0]
        merged = [{"display": a["display"], "subs": a["subs"] + b["subs"], "fields": {**a["fields"], **b["fields"]},
                   "extra_params": {n.label: b["display"]}}
                  for a, b in zip(p.options, n.options)]
        pair = _Axis(p.label, "prompt_pair", None, merged, p.nodes + n.nodes, p.binding_id, False)
        out = [a for a in axes if a is not p and a is not n]
        out.append(pair)
        return out
    return axes


def order_axes(prompt, axes, anc_set):
    """Loader axes outermost, prompts next, sampler params innermost: sort by how many nodes of the
    sub-graph depend on the axis' target nodes (descending), loaders first on ties. itertools.product
    then keeps loader values adjacent, which is what ComfyUI's output cache needs."""
    desc = graph_descendants(prompt, anc_set)
    def key(item):
        i, a = item
        d = max([desc.get(n, 0) for n in a.nodes] or [0])
        return (-d, 0 if a.loader else 1, i)
    return [a for _, a in sorted(enumerate(axes), key=key)]


def normalize_output(entry, prompt, class_info=None):
    """(node_id, output_index, media, frames_from) of the socket to collect. media is "image" or "video"."""
    wo = entry.get("workflow_output")
    if not (isinstance(wo, dict) and wo.get("node_id") is not None):
        wo = default_output(prompt, class_info)
        if not wo:
            raise ValueError("No image or video output found. Pick the socket to collect in the workflow section.")
    nid, idx = str(wo["node_id"]), int(wo.get("output_index", 0) or 0)
    media = str(wo.get("media") or "image").lower()
    if media not in ("image", "video"):
        raise ValueError(f"Unknown output media '{media}' (expected image or video).")
    frames_from = wo.get("frames_from")
    frames_from = None if (frames_from is None or media != "video") else str(frames_from)
    if nid not in prompt:
        raise ValueError(f"The chosen output node {nid} is not in the workflow.")
    if frames_from is not None:
        if frames_from not in prompt:
            raise ValueError(f"The video frames node {frames_from} is not in the workflow.")
        why = _vhs_unsupported(prompt[frames_from])
        if why:
            raise ValueError(f"{node_title(prompt, frames_from)} (#{frames_from}) cannot be used as a video source: {why}.")
    return nid, idx, media, frames_from


def plan_workflow_entry(entry, extra_seeds=None, pos_prompts=None, neg_prompts=None, class_info=None):
    """
    Expand one `model_type: "workflow"` config into jobs.

    Returns {"jobs": [conf, ...], "axes": [{label, count, order}], "warnings": [...],
             "template": {"hash", "prompt", "output", "run_other_outputs"}}.
    Each job conf is flat (4.3 in the plan) and carries wf_subs; the template is NOT copied into confs.
    """
    prompt = parse_api_workflow(entry.get("workflow"))
    out_id, out_idx, media, frames_from = normalize_output(entry, prompt, class_info)
    run_other = bool(entry.get("workflow_run_other_outputs", False))
    warnings = []
    anchor = frames_from or out_id  # frames mode: the VHS node, so its own inputs (frame_rate) count as upstream
    anc = ancestors_ordered(prompt, anchor)
    anc_set = set(anc)
    bindings = entry.get("workflow_bindings") or []

    axes = _build_axes(prompt, bindings, entry, anc_set, extra_seeds, pos_prompts, neg_prompts, class_info, warnings)
    if extra_seeds and not any(a.field == "seed" for a in axes):
        seed_inputs = [f"{node_title(prompt, n)}.{i}" for n in anc for i, v in (prompt[n].get("inputs") or {}).items()
                       if ROLE_HINTS.get(i) == "seed" and not is_link(v)]
        warnings.append("Random extra seeds were ignored: no axis is bound to a seed input. Detected seed inputs: "
                        + (", ".join(seed_inputs) or "none") + ".")
    axes = order_axes(prompt, _pair_prompt_axes(axes), anc_set)

    total = 1
    for a in axes:
        total *= len(a.options)
        if total > MAX_JOBS:
            raise ValueError(f"This sweep would create more than {MAX_JOBS} jobs; narrow the axes.")

    sources, encoders, lora_nodes = _field_sources(prompt, anchor)
    thash = template_hash(prompt, (out_id, out_idx), run_other, media, frames_from)
    source_name = str(entry.get("workflow_source") or "")
    base = {"model_type": "workflow", "media_type": media, "wf_template_hash": thash,
            "wf_source": source_name, "wf_output": f"{out_id}:{out_idx}"}
    if media == "video":
        base["wf_output_media"] = "video"
        if frames_from is not None:
            base["wf_frames_from"] = frames_from

    jobs = []
    clamped_fps = []  # raw out-of-range literal fps values of a frame sink (a swept fps can give several)
    for combo in itertools.product(*[range(len(a.options)) for a in axes]):
        subs, fields, params = [], {}, {}
        for a, oi in zip(axes, combo):
            opt = a.options[oi]
            subs.extend(opt["subs"])
            fields.update(opt["fields"])
            params[a.label] = opt["display"]
            params.update(opt.get("extra_params") or {})
        substituted = apply_subs(prompt, subs)
        if frames_from is not None and frames_from in substituted:
            _fps, _was_clamped = _sink_fps(substituted[frames_from])
            if _was_clamped:
                _raw = _sink_fps_raw(substituted[frames_from])
                if _raw not in clamped_fps:
                    clamped_fps.append(_raw)
        conf = dict(base)
        conf.update(known_fields(substituted, sources, encoders, lora_nodes))
        conf.update(fields)
        conf["positive"] = conf.get("positive", "") if isinstance(conf.get("positive", ""), str) else ""
        conf["negative"] = conf.get("negative", "") if isinstance(conf.get("negative", ""), str) else ""
        conf["wf_subs"] = [[n, i, v] for n, i, v in subs]
        conf["wf_params"] = params
        conf["wf_fingerprint"] = fingerprint(substituted, (out_id, out_idx), media, frames_from)
        jobs.append(conf)

    if clamped_fps:
        lo, hi = CREATE_VIDEO_FPS
        used = sorted({min(max(f, lo), hi) for f in clamped_fps})
        warnings.append(f"{node_title(prompt, frames_from)} (#{frames_from}) fps "
                        f"{', '.join(f'{f:g}' for f in clamped_fps)} "
                        f"{'is' if len(clamped_fps) == 1 else 'are'} outside CreateVideo's 1–120; "
                        f"the collected mp4 uses {' / '.join(f'{u:g}' for u in used)}.")
    return {
        "jobs": jobs,
        "axes": [{"label": a.label, "count": len(a.options), "order": i} for i, a in enumerate(axes)],
        "warnings": warnings,
        "template": {"hash": thash, "prompt": prompt, "output": [out_id, out_idx], "run_other_outputs": run_other,
                     "media": media, "frames_from": frames_from},
    }


def expand_workflow_entry(entry, extra_seeds=None, pos_prompts=None, neg_prompts=None):
    """`config_utils.expand_configs` hook: the expanded job confs of one workflow entry."""
    return plan_workflow_entry(entry, extra_seeds, pos_prompts, neg_prompts)["jobs"]


def plan_workflow_configs(configs, extra_seeds=None, pos_prompts=None, neg_prompts=None, class_info=None):
    """Plan every workflow entry of a configs list (Builder order). Raises ValueError on anything that is
    not a workflow entry. Returns {"jobs", "templates": {hash: {...}}, "axes", "warnings"}."""
    jobs, templates, axes, warnings = [], {}, [], []
    for ci, entry in enumerate(configs):
        if not isinstance(entry, dict) or entry.get("model_type") != "workflow":
            raise ValueError("Workflow sweeps cannot be mixed with standard configs in one sweep plan "
                             "(queue the Generator: it runs the standard configs first, then the sweep).")
        plan = plan_workflow_entry(entry, extra_seeds, pos_prompts, neg_prompts, class_info)
        jobs.extend(plan["jobs"])
        templates[plan["template"]["hash"]] = plan["template"]
        axes.append({"config": entry.get("name", f"Config {ci + 1}"), "axes": plan["axes"]})
        warnings.extend(plan["warnings"])
    return {"jobs": jobs, "templates": templates, "axes": axes, "warnings": warnings}


# --------------------------------------------------------------------------- session / manifest helpers

def session_paths(session_name):
    """Same layout as generation_orchestrator.setup_session_directories (replicated: the orchestrator
    is not imported here). The name is sanitised with the same rule as every other route."""
    import folder_paths
    session = _cu().sanitize_session_name(str(session_name or ""))
    base = os.path.join(folder_paths.get_output_directory(), "benchmarks", session)
    return {"session": session, "base": base, "images": os.path.join(base, "images"),
            "manifest": os.path.join(base, "manifest.json"),
            "sweeps": os.path.join(base, "workflow_sweeps")}


def read_manifest_items(manifest_path):
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"items": []}
    except (OSError, ValueError):
        return {"items": []}


def build_collect_item(conf, width, height, duration, batch_idx, gen_index, create_image_metadata=None):
    """Manifest item for one collected image (without id/file; the collector adds those).
    Uses image_generation.create_image_metadata so the field set matches standard runs."""
    if create_image_metadata is None:
        try:
            from .image_generation import create_image_metadata
        except ImportError:
            from image_generation import create_image_metadata
    conf = dict(conf or {})
    seed = conf.get("seed", 0)
    seed = 0 if seed is None else seed
    meta = create_image_metadata(conf, int(width), int(height), duration, seed, batch_idx,
                                 conf.get("positive", ""), conf.get("negative", ""), gen_index=gen_index)
    if meta.get("denoise") == 1.0:
        meta["denoise"] = 1
    meta.setdefault("model_type", "workflow")
    meta.update({"favorited": False, "rejected": False, "note": ""})
    return meta


def new_item_id():
    return int(time.time() * 100000) + random.randint(0, 1000)


def parse_for_ui(api_json, class_info=comfy_class_info):
    """Response body of POST /config_tester/sweep/parse. Raises ValueError for unusable input."""
    prompt = parse_api_workflow(api_json)
    sources = find_image_sources(prompt, class_info)
    warnings = []
    video_sources = find_video_sources(prompt, class_info, warnings)
    default = default_output(prompt, class_info)
    anchor = {"node_id": default.get("frames_from") or default["node_id"]} if default else None
    suggestions = suggest_bindings(prompt, anchor, class_info) if default else []
    nodes_ = describe_nodes(prompt, class_info)
    missing = sorted({n["class_type"] for n in nodes_ if n["missing_class"]})
    if missing:
        warnings.append("These node types are not installed in this ComfyUI: " + ", ".join(missing))
    if not sources and not video_sources:
        warnings.append("No image or video output was found; pick the socket to collect manually.")
    return {"ok": True, "nodes": nodes_, "outputs": find_output_nodes(prompt, class_info),
            "image_sources": sources, "video_sources": video_sources, "default_output": default,
            "suggestions": suggestions, "warnings": warnings, "errors": []}
