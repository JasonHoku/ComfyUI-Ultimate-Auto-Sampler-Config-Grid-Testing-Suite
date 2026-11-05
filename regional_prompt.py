"""
Regional prompting with COUPLE (GitHub issue #11).

Syntax:
    [<base>] COUPLE[(<layout>)] <region1> | <region2> [| <region3> [| <region4>]]

The prompt is split into a base (applies everywhere) and 2-4 equal-sized regions
(columns left to right, or rows top to bottom with COUPLE(rows)). Each region may
end with a weight ` :w`. Parsing and the join helpers are pure Python; torch is
only imported inside the mask functions.

Conditioning produced from a COUPLE prompt carries a small JSON-safe tag
(REGION_KEY) on each region entry. The tag is size independent, so it survives
the conditioning cache and the distributed serializer. At sample time
``apply_regional_masks`` turns the tags into ComfyUI masks sized to the actual
latent (optionally restricted to a ``frame`` for tiles and crops).
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

REGION_KEY = "uscg_region"
MAX_REGIONS = 4

_AXIS_ALIASES = {
    "cols": "cols", "columns": "cols", "v": "cols", "vertical": "cols",
    "rows": "rows", "h": "rows", "horizontal": "rows",
}

# COUPLE as its own word: start of string or after whitespace, followed by whitespace, "(" or end.
_MARKER_RE = re.compile(r'(?<!\S)COUPLE(?=\s|\(|$)')
_WEIGHT_RE = re.compile(r'^(?:(.*?)\s+)?:(\d+\.?\d*)$', re.DOTALL)
_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSERS = set(_BRACKETS.values())


class RegionalPromptError(ValueError):
    """Invalid COUPLE syntax."""


@dataclass(frozen=True)
class Region:
    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class RegionalSpec:
    base: str
    axis: str
    regions: tuple


# ------------------------------------------------------------------ parsing

def split_at_marker(prompt) -> Optional[Tuple[str, str, str]]:
    """Raw split by position: (base_raw, marker_raw, regions_raw), or None without a marker.

    marker_raw is "COUPLE" or "COUPLE(<layout>)" (an unclosed "(" runs to the end).
    No validation and no cleaning is done here.
    """
    if not isinstance(prompt, str):
        return None
    m = _MARKER_RE.search(prompt)
    if not m:
        return None
    start, end = m.start(), m.end()
    if end < len(prompt) and prompt[end] == "(":
        close = prompt.find(")", end)
        end = len(prompt) if close < 0 else close + 1
    return prompt[:start], prompt[start:end], prompt[end:]


def has_regions(prompt) -> bool:
    """True when the prompt contains the COUPLE keyword."""
    return isinstance(prompt, str) and _MARKER_RE.search(prompt) is not None


def _split_top_level(text, sep="|"):
    """Split on ``sep`` outside of () [] {} <> (bracket depth tracked; stray closers never go negative)."""
    parts = []
    depth = 0
    cur = []
    for ch in text:
        if ch in _BRACKETS:
            depth += 1
        elif ch in _CLOSERS:
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _clean(text, strip_trailing_break=False, strip_leading_break=False):
    """Trim whitespace and commas at both ends; optionally drop a dangling BREAK at one end."""
    s = text.strip(" \t\r\n,")
    while True:
        changed = False
        if strip_trailing_break and s.endswith("BREAK"):
            head = s[:-5]
            if head == "" or head[-1].isspace() or head[-1] == ",":
                s = head.strip(" \t\r\n,")
                changed = True
        if strip_leading_break and s.startswith("BREAK"):
            tail = s[5:]
            if tail == "" or tail[0].isspace() or tail[0] == ",":
                s = tail.strip(" \t\r\n,")
                changed = True
        if not changed:
            return s


def _parse_layout(marker_raw):
    arg = marker_raw[len("COUPLE"):]
    if not arg:
        return "cols"
    if not arg.endswith(")"):
        raise RegionalPromptError("COUPLE( is missing its closing )")
    inner = arg[1:-1].strip()
    if inner == "":
        return "cols"
    if "," in inner:
        raise RegionalPromptError(
            f"COUPLE({inner}): extra arguments are not supported yet (only a layout: cols or rows)")
    axis = _AXIS_ALIASES.get(inner.lower())
    if axis is None:
        raise RegionalPromptError(f"COUPLE({inner}): unknown layout (use cols or rows)")
    return axis


def parse_regional_prompt(prompt) -> Optional[RegionalSpec]:
    """-> RegionalSpec, or None when the prompt has no COUPLE marker. Raises RegionalPromptError on bad syntax."""
    parts = split_at_marker(prompt)
    if parts is None:
        return None
    base_raw, marker_raw, regions_raw = parts
    if _MARKER_RE.search(base_raw) or _MARKER_RE.search(regions_raw):
        raise RegionalPromptError("only one COUPLE is allowed per prompt")
    axis = _parse_layout(marker_raw)

    regions = []
    for raw in _split_top_level(regions_raw, "|"):
        raw = raw.strip()
        weight = 1.0
        wm = _WEIGHT_RE.match(raw)
        if wm:
            weight = float(wm.group(2))
            raw = wm.group(1) or ""
        regions.append(Region(_clean(raw, strip_leading_break=True), weight))

    n = len(regions)
    if n < 2:
        raise RegionalPromptError("COUPLE needs at least 2 regions separated by | (found 1)")
    if n > MAX_REGIONS:
        raise RegionalPromptError(f"COUPLE supports at most {MAX_REGIONS} regions (found {n})")

    base = _clean(base_raw, strip_trailing_break=True)
    active = [r for r in regions if r.text and r.weight > 0]
    if not active:
        raise RegionalPromptError("COUPLE: every region is empty or has weight 0")
    if not base and len(active) != n:
        raise RegionalPromptError(
            "COUPLE: a region is empty or :0 and there is no base text, so part of the image would have no prompt")
    return RegionalSpec(base=base, axis=axis, regions=tuple(regions))


def validate_regional_prompts(prompts):
    """Raise ONE RegionalPromptError listing every invalid COUPLE prompt (cut to 80 chars)."""
    if prompts is None:
        return
    items = sorted(prompts, key=str) if isinstance(prompts, (set, frozenset)) else list(prompts)
    problems = []
    for p in items:
        if not has_regions(p):
            continue
        try:
            parse_regional_prompt(p)
        except RegionalPromptError as e:
            shown = p if len(p) <= 80 else p[:77] + "..."
            problems.append(f"  - {shown!r}: {e}")
    if problems:
        raise RegionalPromptError(
            f"Invalid COUPLE regional prompt(s) ({len(problems)}):\n" + "\n".join(problems))


# ------------------------------------------------------------- join helpers

def join_prompt_parts(parts, body_index, sep=", "):
    """``sep.join(parts)``, except that a COUPLE prompt at ``parts[body_index]`` keeps its
    marker + regions at the END and the other parts land in its base.

    Without COUPLE this returns exactly ``sep.join(parts)`` (cache keys depend on that).
    """
    parts = list(parts)
    if not (0 <= body_index < len(parts)):
        return sep.join(parts)
    split = split_at_marker(parts[body_index])
    if split is None:
        return sep.join(parts)
    base_raw, marker_raw, regions_raw = split
    base = _clean(base_raw, strip_trailing_break=True)
    new_parts = parts[:body_index] + ([base] if base else []) + parts[body_index + 1:]
    joined = sep.join(new_parts)
    tail = marker_raw + " " + regions_raw.strip()
    return (joined + " " + tail) if joined else tail


def append_to_base(prompt, text, sep=" "):
    """``prompt + " " + text`` (sep), but into the base when the prompt has COUPLE regions."""
    return join_prompt_parts([prompt, text], 0, sep)


def plain_prompt_text(prompt):
    """Prompt without the COUPLE marker/weights: base and regions joined with ", " (for scorers)."""
    try:
        spec = parse_regional_prompt(prompt)
    except RegionalPromptError:
        return prompt
    if spec is None:
        return prompt
    texts = ([spec.base] if spec.base else []) + [r.text for r in spec.regions if r.text]
    return ", ".join(texts)


# ------------------------------------------------------------------- masks

def region_tag(index, count, axis, weight=1.0):
    """JSON-safe tag stored on a region's conditioning entries."""
    return {"v": 1, "index": int(index), "count": int(count), "axis": str(axis), "weight": float(weight)}


def build_region_masks(count, axis, height, width, frame=None, device=None):
    """Hard 0/1 masks, float32 (count, height, width); they sum to exactly 1 in every cell.

    ``frame`` = (y0, x0, y1, x1) as fractions of the full image that this latent shows
    (default the whole image). A cell goes to region floor(u * count) where u is the
    cell-centre position along the split axis inside the full image.
    """
    import torch
    y0, x0, y1, x1 = (0.0, 0.0, 1.0, 1.0) if frame is None else (float(v) for v in frame)
    height, width = int(height), int(width)
    if axis == "rows":
        n, f0, f1 = height, y0, y1
    else:
        n, f0, f1 = width, x0, x1
    cells = torch.arange(n, dtype=torch.float64)
    u = f0 + (cells + 0.5) / max(n, 1) * (f1 - f0)
    idx = torch.clamp(torch.floor(u * count).to(torch.int64), 0, count - 1)
    masks = torch.zeros((count, height, width), dtype=torch.float32)
    for k in range(count):
        line = (idx == k).to(torch.float32)
        if axis == "rows":
            masks[k] = line.view(height, 1).expand(height, width)
        else:
            masks[k] = line.view(1, width).expand(height, width)
    if device is not None:
        masks = masks.to(device)
    return masks


def apply_regional_masks(conditioning, latent_shape, frame=None):
    """Turn region tags into ComfyUI masks sized to the actual latent.

    Returns the input object unchanged when no entry carries a tag. Cached dicts are
    never modified. Tagged entries whose mask is empty inside ``frame`` are dropped.
    """
    if not conditioning:
        return conditioning
    try:
        tagged = any(isinstance(e[1], dict) and REGION_KEY in e[1] for e in conditioning)
    except Exception:
        return conditioning
    if not tagged:
        return conditioning

    shape = tuple(int(s) for s in latent_shape)
    h, w = shape[-2], shape[-1]
    frames = shape[-3] if len(shape) >= 5 else None

    mask_cache = {}
    out = []
    counts = set()
    axes = set()
    for entry in conditioning:
        d = entry[1] if len(entry) > 1 else None
        if not (isinstance(d, dict) and REGION_KEY in d):
            out.append(entry)
            continue
        tag = d[REGION_KEY]
        new_d = {k: v for k, v in d.items() if k != REGION_KEY}
        try:
            count, axis = int(tag["count"]), str(tag["axis"])
            index, weight = int(tag["index"]), float(tag.get("weight", 1.0))
            if count < 1 or not 0 <= index < count:
                raise ValueError("bad region index")
        except Exception:
            out.append([entry[0], new_d])  # unreadable tag: behave like an untagged entry
            continue
        key = (count, axis)
        if key not in mask_cache:
            mask_cache[key] = build_region_masks(count, axis, h, w, frame)
        m = mask_cache[key][index]
        if not bool(m.any()):
            continue  # this region is outside the tile/crop: skip its UNet pass
        m = m.unsqueeze(0)
        if frames is not None:
            m = m.unsqueeze(1).expand(1, frames, h, w).contiguous()
        new_d["mask"] = m
        new_d["mask_strength"] = weight
        new_d["set_area_to_bounds"] = False
        counts.add(count)
        axes.add(axis)
        out.append([entry[0], new_d])

    n_txt = "/".join(str(c) for c in sorted(counts)) or "0"
    frame_txt = "" if frame is None else ", frame " + "(" + ", ".join(f"{float(v):.3f}" for v in frame) + ")"
    print(f"[GridTester] \U0001f9e9 Regional prompt: {n_txt} regions ({'/'.join(sorted(axes))}) on {w}x{h} latent{frame_txt}")
    return out
