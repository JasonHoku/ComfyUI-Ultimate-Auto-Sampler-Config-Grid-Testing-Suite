"""
Manifest File Management
Handles saving and merging manifest.json with user modifications
"""

import hashlib
import json
import os
import re
import threading
import time

# Thread-safe lock for manifest file writes.
# Required when multiple distributed workers submit results concurrently.
_manifest_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pairwise review votes (manifest top-level "comparisons" list)
# ---------------------------------------------------------------------------
_WINNERS = ("a", "b", "tie")


def comparison_key(c):
    """Dedupe key of a vote: "ts:a:b" (string)."""
    return f"{c.get('ts')}:{c.get('a')}:{c.get('b')}"


def validate_comparison(c):
    """Return a cleaned vote dict, or None when the entry is not a valid vote."""
    if not isinstance(c, dict):
        return None
    a, b = c.get("a"), c.get("b")
    if type(a) is not int or type(b) is not int or a == b:
        return None
    if c.get("winner") not in _WINNERS:
        return None
    ts = c.get("ts")
    if type(ts) not in (int, float) or ts != ts or ts < 0:
        return None
    mode = c.get("mode")
    mode = mode[:32] if isinstance(mode, str) and mode else "matched"
    out = {"a": a, "b": b, "winner": c["winner"], "ts": ts, "mode": mode}
    if isinstance(c.get("blind"), bool):
        out["blind"] = c["blind"]
    return out


def merge_comparison_lists(existing, append, remove_keys):
    """
    Merge votes: keep existing order, drop entries whose key is in remove_keys,
    append valid new entries whose key is not present yet.
    Invalid existing entries are kept untouched. Returns (merged, added, removed).
    """
    remove = set(remove_keys or ())
    merged = []
    seen = set()
    removed = 0
    for c in (existing or []):
        if isinstance(c, dict) and validate_comparison(c) is not None:
            k = comparison_key(c)
            if k in remove:
                removed += 1
                continue
            seen.add(k)
        merged.append(c)
    added = 0
    for c in (append or []):
        v = validate_comparison(c)
        if v is None:
            continue
        k = comparison_key(v)
        if k in seen or k in remove:
            continue
        seen.add(k)
        merged.append(v)
        added += 1
    return merged, added, removed


def preserve_disk_comparisons(disk_manifest, target_manifest):
    """
    Copy the on-disk votes onto target_manifest (disk wins). Only the compare
    routes create or remove votes, so a stale in-memory copy must never
    resurrect undone/cleared votes or drop new ones.
    """
    try:
        if (isinstance(disk_manifest, dict) and "comparisons" in disk_manifest
                and isinstance(target_manifest, dict)):
            target_manifest["comparisons"] = disk_manifest["comparisons"]
    except Exception:
        pass


def _atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    last = None
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:  # Windows: a reader has the file open
            last = e
            time.sleep(0.04)
    # Fall back to a plain in-place write (same discipline as save_manifest)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    try:
        os.remove(tmp)
    except OSError:
        pass


def update_meta(manifest_path, updates):
    """
    Locked read-modify-write of ONLY keys of the manifest's "meta" dict (items written since the
    caller last looked are preserved). Raises on a missing/unreadable manifest; nothing is written then.
    """
    with _manifest_lock:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("manifest is not a JSON object")
        meta = data.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.update(updates)
        data["meta"] = meta
        _atomic_write_json(manifest_path, data)
        return meta


def update_manifest_comparisons(manifest_path, append=(), remove_keys=(), clear=False):
    """
    Locked read-modify-write of ONLY the "comparisons" list.
    Raises FileNotFoundError when the manifest is missing and ValueError/JSON
    errors when it is unreadable (nothing is written in that case).
    """
    with _manifest_lock:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("manifest is not a JSON object")
        if clear:
            old = data.get("comparisons")
            removed = len(old) if isinstance(old, list) else 0
            data["comparisons"], added = [], 0
        else:
            existing = data.get("comparisons")
            if not isinstance(existing, list):
                existing = []
            data["comparisons"], added, removed = merge_comparison_lists(existing, append, remove_keys)
        _atomic_write_json(manifest_path, data)
        return {"comparisons": data["comparisons"], "added": added,
                "removed": removed, "total": len(data["comparisons"])}


def merge_manifest_user_changes(manifest_path, existing_data):
    """
    Reload manifest and merge user changes (favorites, rejected, notes) to preserve them.
    This prevents losing user modifications when the manifest is saved during generation.
    
    Args:
        manifest_path: Path to the manifest.json file
        existing_data: The current manifest data dictionary to update
    """
    try:
        with open(manifest_path, "r") as f:
            current_manifest = json.load(f)

        # Votes are owned by the compare routes: the disk copy always wins
        preserve_disk_comparisons(current_manifest, existing_data)
        
        # Create lookup dict of current items by ID from DISK
        current_items_dict = {
            item.get("id"): item 
            for item in current_manifest.get("items", []) 
            if "id" in item
        }
        
        # Track merge statistics
        merged_count = 0
        favorites_preserved = 0
        rejected_preserved = 0
        notes_preserved = 0
        
        # Update ALL items in existing_data (both new and old) with user modifications from disk
        for item in existing_data["items"]:
            item_id = item.get("id")
            if item_id and item_id in current_items_dict:
                current_item = current_items_dict[item_id]
                merged_count += 1
                
                # Preserve user-modified fields from the version on disk
                # Field names: "favorited" (not "favorite"), "rejected", "note" (not "notes")
                if current_item.get("favorited"):
                    item["favorited"] = True
                    favorites_preserved += 1

                if current_item.get("rejected"):
                    item["rejected"] = True
                    rejected_preserved += 1

                if current_item.get("note"):
                    item["note"] = current_item["note"]
                    notes_preserved += 1
                # Per-image metrics computed for an existing session (metrics_runner) live only on
                # disk; a generation run's in-memory copy must not wipe them on its next save.
                if "metrics" in current_item and "metrics" not in item:
                    item["metrics"] = current_item["metrics"]
            # else: This is a new item, no merge needed
        
        # Log merge results if any user changes were preserved
        if favorites_preserved > 0 or rejected_preserved > 0 or notes_preserved > 0:
            print(f"[GridTester] 🔄 Merged user changes: {favorites_preserved} favorites, {rejected_preserved} rejected, {notes_preserved} notes (from {merged_count} items)")
                    
    except FileNotFoundError:
        # First save, no existing manifest to merge
        pass
    except Exception as e:
        print(f"[GridTester] ⚠️ Warning: Could not merge manifest changes: {e}")
        import traceback
        traceback.print_exc()


def save_manifest(manifest_path, manifest_data):
    """
    Save manifest data to disk, merging user changes first.
    Thread-safe: uses _manifest_lock for concurrent access from distributed workers.

    Args:
        manifest_path: Path to the manifest.json file
        manifest_data: Manifest data dictionary to save
    """
    with _manifest_lock:
        # Merge user changes from disk first
        merge_manifest_user_changes(manifest_path, manifest_data)

        # Save merged data
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

        print(f"[GridTester] 💾 Manifest saved: {manifest_path}")


def load_existing_manifest(manifest_path):
    """
    Load existing manifest data from disk.
    
    Args:
        manifest_path: Path to the manifest.json file
        
    Returns:
        dict: Manifest data or new empty manifest structure
    """
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            print(f"[GridTester] 📂 Loaded existing manifest with {len(data.get('items', []))} items")
            return data
        except Exception as e:
            print(f"[GridTester] ⚠️ Error loading manifest: {e}")
    
    # Return new manifest structure
    return {
        "session_name": "default",
        "items": []
    }



def _item_file_path(manifest_path, item):
    """Local path of an item's image, derived from its /view URL, confined to <session>/images."""
    try:
        m = re.search(r"[?&]filename=([^&]+)", str(item.get("file", "")))
        filename = os.path.basename(m.group(1)) if m else ""
        if not filename:
            return None
        images_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "images")
        path = os.path.abspath(os.path.join(images_dir, filename))
        if os.path.dirname(path) != os.path.abspath(images_dir):
            return None
        return path
    except Exception:
        return None


def append_items(manifest_path, items, remove_ids=(), session_name=None, default_meta=None):
    """
    Add items to a manifest (newest first, like every other writer) with a fresh read of the file on disk.

    Used by workflow sweeps, where a standard Generator run on the same session may write the manifest
    between two jobs, so an in-memory copy held elsewhere would drop its items. Thread-safe.
    `remove_ids` are deleted (item + image file) after the new items are inserted (overwrite_existing).
    Returns the manifest dict that was written.
    """
    remove_ids = set(remove_ids or ())
    with _manifest_lock:
        data = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[GridTester] ⚠️ Could not read manifest before append: {e}")
        if not isinstance(data, dict):
            data = {"session_name": session_name or "default", "items": []}
        data.setdefault("items", [])
        if session_name:
            data.setdefault("session_name", session_name)
        if default_meta and not data.get("meta"):
            data["meta"] = default_meta
        for item in items:
            data["items"].insert(0, item)
        if remove_ids:
            kept = []
            for it in data["items"]:
                if it.get("id") in remove_ids and not any(it is n for n in items):
                    path = _item_file_path(manifest_path, it)
                    if path and os.path.isfile(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                else:
                    kept.append(it)
            data["items"] = kept
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, manifest_path)
        return data


def base_dims_of(item):
    """{"base_width","base_height"} of the generated image an upscale of `item` starts from ({} if unknown)."""
    bw, bh = item.get("base_width"), item.get("base_height")
    if bw and bh:
        return {"base_width": bw, "base_height": bh}
    if item.get("upscaled"):
        return {}          # legacy upscaled source: let normResolution estimate
    w, h = item.get("width"), item.get("height")
    return {"base_width": w, "base_height": h} if w and h else {}


# ---------------------------------------------------------------------------
# Stable ids for scanned items (a re-scan of the same folder keeps ids, so votes,
# favorites, notes and metrics stay attached)
# ---------------------------------------------------------------------------
SCAN_ID_BASE = 1 << 52          # scanned ids live in [2^52, 2^53); generated ids are ~1.8e14 (time*1e5)


def stable_item_id(key, salt=0):
    """Deterministic int id in [2^52, 2^53) for `key` (JS-safe: < Number.MAX_SAFE_INTEGER + 1)."""
    k = key if not salt else f"{key}#{salt}"
    h = int.from_bytes(hashlib.blake2b(k.encode("utf-8"), digest_size=8).digest(), "big") >> 12   # 52 bits
    return SCAN_ID_BASE + h


def scan_relpath(file_path, directory_path):
    """Path of a scanned file relative to the scanned directory, with forward slashes."""
    return os.path.relpath(file_path, directory_path).replace(os.sep, "/")


def assign_scan_ids(items, reserved=()):
    """
    Give every item a stable id derived from its `source_relpath`, in place. Items are processed
    sorted by relpath (not mtime, which changes), `reserved` ids are never reused, and a clash is
    resolved deterministically by retrying with salt 1, 2, ... so the same files get the same ids
    on every machine and scan. Items without a relpath are left alone (their id is reserved).
    """
    used = set(reserved)
    todo = []
    for it in items:
        if it.get("source_relpath"):
            todo.append(it)
        elif "id" in it:
            used.add(it["id"])
    for it in sorted(todo, key=lambda x: x["source_relpath"]):
        salt = 0
        while True:
            cid = stable_item_id("scan:" + it["source_relpath"], salt)
            if cid not in used:
                break
            salt += 1
        it["id"] = cid
        used.add(cid)
    return items


def merge_rescan(old, new):
    """
    Merge a fresh scan manifest (`new`, modified in place and returned) with the manifest already on
    disk for the same session: an old item matched by `source_relpath` (else legacy `source_file`)
    donates its id (one-time migration of random legacy ids) and its user fields (favorited, rejected,
    note, notes, metrics); top-level `comparisons` and meta.rejected_preview_fps are carried over.
    """
    if not isinstance(old, dict):
        return new
    old_items = [i for i in (old.get("items") or []) if isinstance(i, dict)]
    by_rel, by_src = {}, {}
    for oi in old_items:
        if oi.get("source_relpath"):
            by_rel.setdefault(oi["source_relpath"], oi)
        if oi.get("source_file"):
            by_src.setdefault(oi["source_file"], oi)
    reused, pending = set(), []
    for it in new.get("items", []):
        oi = by_rel.get(it.get("source_relpath")) or by_src.get(it.get("source_file"))
        if oi is None:
            pending.append(it)
            continue
        if "id" in oi:
            it["id"] = oi["id"]
            reused.add(oi["id"])
        it["favorited"] = bool(oi.get("favorited", False))
        it["rejected"] = bool(oi.get("rejected", False))
        for k in ("note", "notes"):
            if oi.get(k):
                it[k] = oi[k]
        if "metrics" in oi and "metrics" not in it:
            it["metrics"] = oi["metrics"]
    # new files: ids stay stable unless they clash with a reused legacy id
    fresh = [i for i in pending if isinstance(i.get("id"), int) and i["id"] >= SCAN_ID_BASE and i.get("source_relpath")]
    keep = {i["id"] for i in pending if i not in fresh and "id" in i}
    assign_scan_ids(fresh, reserved=reused | keep)
    if isinstance(old.get("comparisons"), list):
        new["comparisons"] = old["comparisons"]
    old_meta = old.get("meta")
    if isinstance(old_meta, dict) and old_meta.get("rejected_preview_fps"):
        new.setdefault("meta", {})["rejected_preview_fps"] = old_meta["rejected_preview_fps"]
    return new


def write_rescan_manifest(manifest_path, manifest):
    """Locked merge of a re-scan into the manifest on disk (if any) + atomic write. Returns the manifest."""
    with _manifest_lock:
        old = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    old = json.load(f)
            except Exception as e:
                print(f"[GridTester] ⚠️ Could not merge old manifest on re-scan: {e}")
        merge_rescan(old, manifest)
        _atomic_write_json(manifest_path, manifest)
    return manifest
