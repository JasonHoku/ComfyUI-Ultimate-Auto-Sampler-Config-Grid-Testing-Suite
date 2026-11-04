"""
Directory Scanner - Scan arbitrary directories for images and generate manifests.
Parses A1111-compatible PNG metadata into dashboard-compatible manifest items.
"""

import os
import re
import json
import time
import random
from PIL import Image

from .metadata_packer import extract_metadata_from_image
from .manifest_utils import stable_item_id, scan_relpath, assign_scan_ids


# =============================================================================
# A1111 SAMPLER NAME LOOKUP
# =============================================================================

def _build_sampler_lookup():
    """
    Build a mapping from A1111 sampler display names to ComfyUI (sampler, scheduler) tuples.
    Covers common samplers used by A1111/Forge/ComfyUI.
    """
    lookup = {}

    # Base samplers (no scheduler suffix = "normal")
    base_samplers = {
        "Euler": "euler",
        "Euler a": "euler_ancestral",
        "Heun": "heun",
        "DPM2": "dpm_2",
        "DPM2 a": "dpm_2_ancestral",
        "LMS": "lms",
        "DPM++ 2S a": "dpmpp_2s_ancestral",
        "DPM++ 2M": "dpmpp_2m",
        "DPM++ SDE": "dpmpp_sde",
        "DPM++ 2M SDE": "dpmpp_2m_sde",
        "DPM++ 3M SDE": "dpmpp_3m_sde",
        "DPM fast": "dpm_fast",
        "DPM adaptive": "dpm_adaptive",
        "LMS": "lms",
        "DDIM": "ddim",
        "PLMS": "plms",
        "UniPC": "uni_pc",
        "LCM": "lcm",
    }

    schedulers = ["normal", "karras", "exponential", "sgm_uniform"]

    for display_name, comfy_name in base_samplers.items():
        # Without scheduler suffix -> normal
        lookup[display_name.lower()] = (comfy_name, "normal")

        # With scheduler suffixes
        for scheduler in schedulers:
            suffixed = f"{display_name} {scheduler}".lower()
            lookup[suffixed] = (comfy_name, scheduler)

    return lookup


_SAMPLER_LOOKUP = _build_sampler_lookup()


def _parse_sampler_string(sampler_str):
    """
    Parse an A1111-format sampler string into (sampler, scheduler) tuple.
    E.g., "DPM++ 2M Karras" -> ("dpmpp_2m", "karras")
    """
    if not sampler_str:
        return "unknown", "unknown"

    key = sampler_str.strip().lower()

    # Direct lookup
    if key in _SAMPLER_LOOKUP:
        return _SAMPLER_LOOKUP[key]

    # Try stripping known scheduler suffixes
    for scheduler in ["karras", "exponential", "sgm_uniform"]:
        if key.endswith(f" {scheduler}"):
            base = key[: -(len(scheduler) + 1)]
            if base in _SAMPLER_LOOKUP:
                return _SAMPLER_LOOKUP[base][0], scheduler

    # Fallback: return raw string as sampler, normal as scheduler
    return sampler_str.strip(), "normal"


# =============================================================================
# A1111 PARAMETERS STRING PARSER
# =============================================================================

def parse_a1111_parameters(parameters_text):
    """
    Parse an A1111-compatible 'parameters' PNG text chunk into a dict of fields.

    Format:
        Line 1+: positive prompt (may include <lora:name:strength> tags)
        Line N:  "Negative prompt: ..."
        Last line: "Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 1024x1536, Model: name"

    Returns:
        dict with keys: positive, negative, steps, sampler, scheduler, cfg, seed,
                        width, height, model, denoise, lora, clip_skip
    """
    result = {
        "positive": "",
        "negative": "",
        "steps": 0,
        "sampler": "unknown",
        "scheduler": "unknown",
        "cfg": 0,
        "seed": 0,
        "width": 0,
        "height": 0,
        "model": "Unknown",
        "denoise": 1,
        "lora": "None",
        "clip_skip": -1,
    }

    if not parameters_text or not parameters_text.strip():
        return result

    lines = parameters_text.strip().split("\n")
    if not lines:
        return result

    # Find the params line (last line with key-value pairs like "Steps: X, ...")
    params_line = None
    params_line_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r"^\s*Steps:\s*\d+", lines[i]):
            params_line = lines[i]
            params_line_idx = i
            break

    # Find negative prompt line
    neg_line_idx = params_line_idx
    positive_lines = []
    negative_lines = []

    for i, line in enumerate(lines):
        if i >= params_line_idx:
            break
        if line.startswith("Negative prompt:"):
            neg_line_idx = i
            negative_lines.append(line[len("Negative prompt:"):].strip())
        elif i > neg_line_idx:
            negative_lines.append(line)
        else:
            positive_lines.append(line)

    raw_positive = "\n".join(positive_lines).strip()
    result["negative"] = "\n".join(negative_lines).strip()

    # Extract LoRA tags from positive prompt: <lora:name:strength>
    lora_parts = []
    lora_pattern = re.compile(r"<lora:([^:>]+):([^>]+)>")
    for match in lora_pattern.finditer(raw_positive):
        lora_name = match.group(1).strip()
        strength = match.group(2).strip()
        # Parse dual strength if present (strength_model:strength_clip)
        if ":" in strength:
            parts = strength.split(":")
            lora_parts.append(f"{lora_name}.safetensors:{parts[0]}:{parts[1]}")
        else:
            lora_parts.append(f"{lora_name}.safetensors:{strength}:{strength}")

    # Build lora string
    if lora_parts:
        result["lora"] = " + ".join(lora_parts)

    # Strip LoRA tags from displayed positive prompt
    clean_positive = lora_pattern.sub("", raw_positive).strip()
    # Clean up extra spaces and trailing commas
    clean_positive = re.sub(r"\s{2,}", " ", clean_positive).strip().rstrip(",").strip()
    result["positive"] = clean_positive

    # Parse key-value params line
    if params_line:
        # Split by ", " but handle quoted values and JSON
        # Use a regex that handles "Key: Value" pairs
        kv_pattern = re.compile(
            r'(\w[\w\s]*?):\s*("(?:[^"\\]|\\.)*"|{[^}]*}|\[[^\]]*\]|[^,]+)'
        )
        params = {}
        for match in kv_pattern.finditer(params_line):
            key = match.group(1).strip()
            value = match.group(2).strip().strip('"')
            params[key] = value

        # Extract known fields
        if "Steps" in params:
            try:
                result["steps"] = int(params["Steps"])
            except ValueError:
                pass

        if "Sampler" in params:
            sampler, scheduler = _parse_sampler_string(params["Sampler"])
            result["sampler"] = sampler
            result["scheduler"] = scheduler

        if "CFG scale" in params:
            try:
                result["cfg"] = float(params["CFG scale"])
            except ValueError:
                pass

        if "Seed" in params:
            try:
                result["seed"] = int(params["Seed"])
            except ValueError:
                pass

        if "Size" in params:
            size_match = re.match(r"(\d+)x(\d+)", params["Size"])
            if size_match:
                result["width"] = int(size_match.group(1))
                result["height"] = int(size_match.group(2))

        if "Model" in params:
            result["model"] = params["Model"]

        if "Denoising strength" in params:
            try:
                result["denoise"] = float(params["Denoising strength"])
            except ValueError:
                pass

        if "Clip skip" in params:
            try:
                result["clip_skip"] = -abs(int(params["Clip skip"]))
            except ValueError:
                pass

    return result


# =============================================================================
# EXISTING MANIFEST DETECTION
# =============================================================================

def _extract_filename_from_url(file_url):
    """
    Extract the image filename from a manifest item's file URL.
    Handles /view?filename=xxx&... URLs (both generated and external scan).
    """
    if not file_url:
        return None

    # Standard ComfyUI format: /view?filename=img_xxx.webp&type=output&subfolder=...
    match = re.search(r'filename=([^&]+)', file_url)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))

    return None


def _load_existing_manifest_items(directory_path):
    """
    Check if a directory (or its parent) contains a manifest.json generated by this node.
    If found, returns a dict mapping image filename -> item metadata.

    Checks:
      1. {directory_path}/manifest.json  (user scanned the session root)
      2. {directory_path}/../manifest.json  (user scanned the images/ subdirectory)

    Returns:
        tuple: (items_by_filename dict, manifest_meta dict or None)
    """
    candidates = [
        os.path.join(directory_path, "manifest.json"),
    ]
    # If the user is scanning an "images" subdirectory, check the parent for manifest
    if os.path.basename(directory_path).lower() == "images":
        candidates.append(os.path.join(directory_path, "..", "manifest.json"))

    for manifest_path in candidates:
        manifest_path = os.path.normpath(manifest_path)
        if not os.path.isfile(manifest_path):
            continue

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"[DirScanner] Warning: Could not read {manifest_path}: {e}")
            continue

        # Validate it looks like our manifest (must have "items" list)
        items_list = manifest.get("items")
        if not isinstance(items_list, list) or len(items_list) == 0:
            continue

        # Build lookup: filename -> item data
        items_by_filename = {}
        for item in items_list:
            file_url = item.get("file", "")
            filename = _extract_filename_from_url(file_url)
            if filename:
                items_by_filename[filename] = item

        if items_by_filename:
            print(f"[DirScanner] Found existing manifest at {manifest_path} with {len(items_by_filename)} items")
            return items_by_filename, manifest.get("meta")

    return {}, None


# =============================================================================
# DIRECTORY SCANNER
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _probe_mp4_dimensions(file_path):
    """
    Pure-Python MP4/MOV dimension probe. Parses the ISO BMFF container
    to find the first video track's tkhd box and extracts width/height.
    No dependencies — reads only the box headers (a few hundred bytes).

    Returns:
        tuple (width, height) or None if parsing fails or no video track.
    """
    def _find_tkhd(f, end_offset):
        CONTAINER_BOXES = (b'moov', b'trak', b'mdia', b'minf', b'stbl', b'edts')
        while f.tell() < end_offset:
            header = f.read(8)
            if len(header) < 8:
                return None
            size = int.from_bytes(header[0:4], 'big')
            box_type = header[4:8]

            if size == 1:
                # 64-bit largesize
                size_bytes = f.read(8)
                if len(size_bytes) < 8:
                    return None
                size = int.from_bytes(size_bytes, 'big')
                content_size = size - 16
            elif size == 0:
                # Box extends to end of file
                content_size = end_offset - f.tell()
            else:
                content_size = size - 8

            if content_size < 0:
                return None

            content_start = f.tell()
            content_end = content_start + content_size

            if box_type == b'tkhd':
                try:
                    version = f.read(1)[0]
                    f.read(3)  # flags
                    if version == 0:
                        # 32-bit time fields: 4+4+4+4+4 = 20 bytes total
                        f.read(20)
                    else:
                        # 64-bit time fields: 8+8+4+4+8 = 32 bytes
                        f.read(32)
                    f.read(8)  # reserved
                    f.read(8)  # layer, alt_group, volume, reserved
                    f.read(36)  # unity matrix
                    width_raw = int.from_bytes(f.read(4), 'big')
                    height_raw = int.from_bytes(f.read(4), 'big')
                    # Width/height are 16.16 fixed-point
                    width = width_raw >> 16
                    height = height_raw >> 16
                    if width > 0 and height > 0:
                        return (width, height)
                except Exception:
                    pass
                f.seek(content_end)
                continue

            if box_type in CONTAINER_BOXES:
                result = _find_tkhd(f, content_end)
                if result:
                    return result

            f.seek(content_end)
        return None

    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            return _find_tkhd(f, file_size)
    except Exception:
        return None


def scan_directory_for_images(directory_path, max_items=5000, view_subfolder=""):
    """
    Scan a directory for image files and build manifest-compatible items.

    Args:
        directory_path: Absolute path to the directory to scan
        max_items: Maximum number of images to process
        view_subfolder: Subfolder path for ComfyUI's /view endpoint
                        (e.g. "benchmarks/session/external_images")

    Returns:
        tuple: (items_list, stats_dict)
            - items_list: List of manifest-compatible item dicts
            - stats_dict: { "total": N, "with_metadata": M, "skipped": S }
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    items = []
    stats = {"total": 0, "with_metadata": 0, "skipped": 0, "from_manifest": 0}

    # Check for existing manifest.json generated by this node
    manifest_items, manifest_meta = _load_existing_manifest_items(directory_path)

    # Collect image files, sorted by name for consistent ordering
    # Also scan images/ subdirectory if it exists (generated sessions store images there)
    image_files = []
    scan_dirs = [directory_path]
    images_subdir = os.path.join(directory_path, "images")
    if os.path.isdir(images_subdir):
        scan_dirs.append(images_subdir)

    try:
        for scan_dir in scan_dirs:
            for entry in os.scandir(scan_dir):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in MEDIA_EXTENSIONS:
                        try:
                            mtime = entry.stat().st_mtime
                        except Exception:
                            mtime = 0
                        image_files.append((entry.path, mtime))
    except PermissionError:
        raise ValueError(f"Permission denied: {directory_path}")

    # Sort by modification time DESCENDING (newest first) to match the
    # manifest convention used by generated sessions (items are prepended
    # as they're created). The Dashboard's fast-path for oldest/newest sort
    # assumes activeData is stored newest-first.
    image_files.sort(key=lambda x: (-x[1], x[0]))

    if len(image_files) > max_items:
        image_files = image_files[:max_items]

    total_files = len(image_files)
    needs_id = []   # scan-assigned ids: collision-checked together after the loop
    for i, (file_path, _mtime) in enumerate(image_files):
        # gen_index: oldest=0, newest=total-1. Since image_files is
        # newest-first, index 0 is the newest file.
        gen_idx = total_files - 1 - i
        stats["total"] += 1
        filename = os.path.basename(file_path)

        # If we have manifest data for this image, use it (much richer metadata)
        if filename in manifest_items:
            item = manifest_items[filename].copy()
            # Use ComfyUI's built-in /view endpoint with subfolder
            from urllib.parse import quote
            item["file"] = f"/view?filename={quote(filename, safe='')}&type=output&subfolder={quote(view_subfolder, safe='/')}"
            item["source_file"] = filename
            item["source_relpath"] = scan_relpath(file_path, directory_path)
            # Assign gen_index based on mtime-sorted order so Oldest/Newest sort
            # matches file age (overrides any stale gen_index from cached manifest)
            item["gen_index"] = gen_idx
            # Ensure it has an ID
            if "id" not in item:
                item["id"] = stable_item_id("scan:" + item["source_relpath"])
                needs_id.append(item)
            # For videos: ensure media_type is set and probe real dimensions
            # if the cached entry has missing or stale (default 1280x720) values.
            ext_check = os.path.splitext(filename)[1].lower()
            if ext_check in VIDEO_EXTENSIONS:
                item["media_type"] = "video"
                needs_probe = (
                    not item.get("width")
                    or not item.get("height")
                    or (item.get("width") == 1280 and item.get("height") == 720)
                )
                if needs_probe:
                    probed = _probe_mp4_dimensions(file_path)
                    if probed:
                        item["width"], item["height"] = probed
            else:
                if "media_type" not in item:
                    item["media_type"] = "image"
            items.append(item)
            stats["from_manifest"] += 1
            continue

        # No manifest entry — fall through to normal image processing
        try:
            item = _process_single_image(file_path, directory_path, view_subfolder, gen_index=gen_idx)
            if item:
                items.append(item)
                needs_id.append(item)
        except Exception as e:
            print(f"[DirScanner] Warning: Skipping {filename}: {e}")
            stats["skipped"] += 1
            continue

    # Same files -> same ids on every scan; a hash clash is resolved deterministically.
    _scan_assigned = {id(i) for i in needs_id}
    assign_scan_ids(needs_id, reserved={i["id"] for i in items if id(i) not in _scan_assigned and "id" in i})

    # Count items that had metadata
    stats["with_metadata"] = sum(
        1 for item in items if item.get("model", "Unknown") != "Unknown"
    )

    if stats["from_manifest"] > 0:
        print(f"[DirScanner] Used existing manifest for {stats['from_manifest']}/{stats['total']} images")

    return items, stats


def _process_single_image(file_path, directory_path, view_subfolder="", gen_index=0):
    """
    Process a single image or video file into a manifest-compatible item dict.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    # Stable id: derived from the path relative to the scanned folder, so a re-scan keeps it
    relpath = scan_relpath(file_path, directory_path)
    ts = stable_item_id("scan:" + relpath)

    # Build URL using ComfyUI's built-in /view endpoint
    from urllib.parse import quote
    file_url = f"/view?filename={quote(filename, safe='')}&type=output&subfolder={quote(view_subfolder, safe='/')}"

    if is_video:
        # Try to probe real dimensions from MP4/MOV container
        # Falls back to 16:9 default if probing fails (e.g. WebM, corrupt file)
        probed = _probe_mp4_dimensions(file_path)
        vid_w, vid_h = probed if probed else (1280, 720)
        return {
            "id": ts,
            "gen_index": gen_index,
            "file": file_url,
            "media_type": "video",
            "width": vid_w,
            "height": vid_h,
            "seed": 0,
            "steps": 0,
            "cfg": 0,
            "sampler": "video",
            "scheduler": "video",
            "positive": "",
            "negative": "",
            "denoise": 1,
            "model": "Video",
            "lora": "None",
            "clip_skip": -1,
            "duration": 0,
            "batch_idx": 0,
            "rejected": False,
            "favorited": False,
            "source_file": filename,
            "source_relpath": relpath,
        }

    # Get image dimensions from PIL
    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    # Try to extract metadata
    metadata = {}
    parsed = {}
    try:
        metadata = extract_metadata_from_image(file_path)
    except Exception:
        pass

    if metadata.get("parameters"):
        parsed = parse_a1111_parameters(metadata["parameters"])

    # Use parsed metadata, fall back to defaults
    item = {
        "id": ts,
        "gen_index": gen_index,
        "file": file_url,
        "media_type": "image",
        "width": parsed.get("width") or width,
        "height": parsed.get("height") or height,
        "seed": parsed.get("seed", 0),
        "steps": parsed.get("steps", 0),
        "cfg": parsed.get("cfg", 0),
        "sampler": parsed.get("sampler", "unknown"),
        "scheduler": parsed.get("scheduler", "unknown"),
        "positive": parsed.get("positive", ""),
        "negative": parsed.get("negative", ""),
        "denoise": parsed.get("denoise", 1),
        "model": parsed.get("model", "Unknown"),
        "lora": parsed.get("lora", "None"),
        "clip_skip": parsed.get("clip_skip", -1),
        "duration": 0,
        "batch_idx": 0,
        "rejected": False,
        "favorited": False,
        "source_file": filename,
        "source_relpath": relpath,
    }

    return item
