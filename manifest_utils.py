"""
Manifest File Management
Handles saving and merging manifest.json with user modifications
"""

import json
import os
import re
import threading

# Thread-safe lock for manifest file writes.
# Required when multiple distributed workers submit results concurrently.
_manifest_lock = threading.Lock()


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
