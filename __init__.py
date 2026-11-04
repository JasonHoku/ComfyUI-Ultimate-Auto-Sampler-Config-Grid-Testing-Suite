import re
import asyncio
import server
from aiohttp import web
import json
import os
import urllib.parse
import folder_paths
import shutil
import uuid
import time
import inspect
from .sampler_node import SamplerGridTester
from .dashboard_node import SamplerConfigDashboardViewer
from .html_generator import get_html_template
from .config_builder_node import UltimateConfigBuilder
from .json_text_node import SmartJSONTextNode
from .metadata_packer import pack_metadata_into_image
from .directory_scanner import scan_directory_for_images
from .run_api import (
    RunTracker, sanitize_session_name_strict, normalize_configs_payload, apply_run_overrides,
    build_run_prompt, resolve_saved_config, saved_config_to_configs_obj, summarize_history, find_in_queue,
)
from .workflow_sweep_collect import USCGSweepCollect, USCGSweepCollectVideo
import sys

# Register a sys.modules alias for cross-companion imports.
# The Distributed companion (ComfyUI-USCG-Distributed) lazy-imports main USCG
# modules via `from comfyui_uscg_main.<module> import <func>`. The folder name
# 'ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite' has hyphens (not
# valid identifiers), so we publish the underscored alias for Python imports.
sys.modules["comfyui_uscg_main"] = sys.modules[__name__]


# --- PATH SECURITY HELPERS ---
def _get_benchmarks_base():
    """Return the canonical benchmarks base directory."""
    return os.path.realpath(os.path.join(folder_paths.get_output_directory(), "benchmarks"))

def _is_path_within(path, base):
    """Check that *path* is contained within *base* after resolving symlinks."""
    real_path = os.path.realpath(path)
    real_base = os.path.realpath(base)
    return real_path == real_base or real_path.startswith(real_base + os.sep)


# --- CONFIG MANAGEMENT PATH ---
CONFIGS_DIR = os.path.join(folder_paths.get_output_directory(), "ultimate-configs")
os.makedirs(CONFIGS_DIR, exist_ok=True)

# --- CUSTOM RESOLUTIONS FILE ---
CUSTOM_RESOLUTIONS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-custom-resolutions.json")

# --- UPSCALE PRESETS FILE ---
UPSCALE_PRESETS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-upscale-presets.json")

# --- CONFIG SECTION PRESETS FILE ---
CONFIG_SECTION_PRESETS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-config-section-presets.json")

# --- INDIVIDUAL SECTION PRESETS FILES ---
MODELS_PRESETS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-models-presets.json")
LORAS_PRESETS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-loras-presets.json")
PROMPTS_PRESETS_FILE = os.path.join(folder_paths.get_output_directory(), "benchmarks", "USCG-prompts-presets.json")


# =============================================================================
# API: CONFIG MANAGEMENT
# =============================================================================

@server.PromptServer.instance.routes.get("/configbuilder/list_configs")
async def list_configs(request):
    try:
        if not os.path.exists(CONFIGS_DIR):
            os.makedirs(CONFIGS_DIR, exist_ok=True)
        
        files = [f for f in os.listdir(CONFIGS_DIR) if f.endswith(".json")]
        files.sort()
        return web.json_response(files)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.post("/configbuilder/save_config")
async def save_config(request):
    try:
        data = await request.json()
        filename = data.get("name")
        config_data = data.get("data")
        
        if not filename or not config_data:
            return web.Response(status=400, text="Missing name or data")
        
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Sanitize: strip to basename to prevent directory traversal
        filename = os.path.basename(filename)
        filepath = os.path.join(CONFIGS_DIR, filename)

        if not _is_path_within(filepath, CONFIGS_DIR):
            return web.Response(status=403, text="Forbidden")

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=4)
            
        return web.Response(status=200, text="Saved")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving config: {e}")
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/configbuilder/custom_resolutions")
async def get_custom_resolutions(request):
    try:
        if os.path.exists(CUSTOM_RESOLUTIONS_FILE):
            with open(CUSTOM_RESOLUTIONS_FILE, "r") as f:
                data = json.load(f)
            return web.json_response(data)
        return web.json_response({"categories": []})
    except Exception as e:
        return web.json_response({"categories": []})

@server.PromptServer.instance.routes.post("/configbuilder/custom_resolutions")
async def save_custom_resolutions(request):
    try:
        data = await request.json()
        with open(CUSTOM_RESOLUTIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return web.Response(status=200, text="Saved")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving custom resolutions: {e}")
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/configbuilder/upscale_presets")
async def get_upscale_presets(request):
    try:
        if os.path.exists(UPSCALE_PRESETS_FILE):
            with open(UPSCALE_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response(data)
        return web.json_response({"presets": []})
    except Exception as e:
        return web.json_response({"presets": []})

@server.PromptServer.instance.routes.post("/configbuilder/upscale_presets")
async def save_upscale_presets(request):
    try:
        body = await request.text()
        if not body or not body.strip():
            return web.Response(status=400, text="Empty request body")
        data = json.loads(body)
        os.makedirs(os.path.dirname(UPSCALE_PRESETS_FILE), exist_ok=True)
        with open(UPSCALE_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return web.Response(status=200, text="Saved")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving upscale presets: {e}")
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.get("/configbuilder/config_section_presets")
async def get_config_section_presets(request):
    try:
        if os.path.exists(CONFIG_SECTION_PRESETS_FILE):
            with open(CONFIG_SECTION_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response(data)
        return web.json_response({"presets": []})
    except Exception as e:
        return web.json_response({"presets": []})

@server.PromptServer.instance.routes.post("/configbuilder/config_section_presets")
async def save_config_section_presets(request):
    try:
        body = await request.text()
        if not body or not body.strip():
            return web.Response(status=400, text="Empty request body")
        data = json.loads(body)
        os.makedirs(os.path.dirname(CONFIG_SECTION_PRESETS_FILE), exist_ok=True)
        with open(CONFIG_SECTION_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return web.Response(status=200, text="Saved")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving config section presets: {e}")
        return web.Response(status=500, text=str(e))

# --- GENERIC SECTION PRESETS (models, loras, prompts) ---
_SECTION_PRESET_FILES = {
    "models": MODELS_PRESETS_FILE,
    "loras": LORAS_PRESETS_FILE,
    "prompts": PROMPTS_PRESETS_FILE,
}

@server.PromptServer.instance.routes.get("/configbuilder/section_presets")
async def get_section_presets(request):
    section = request.query.get("section", "")
    filepath = _SECTION_PRESET_FILES.get(section)
    if not filepath:
        return web.Response(status=400, text=f"Unknown section: {section}")
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return web.json_response(json.load(f))
        return web.json_response({"presets": []})
    except Exception:
        return web.json_response({"presets": []})

@server.PromptServer.instance.routes.post("/configbuilder/section_presets")
async def save_section_presets(request):
    section = request.query.get("section", "")
    filepath = _SECTION_PRESET_FILES.get(section)
    if not filepath:
        return web.Response(status=400, text=f"Unknown section: {section}")
    try:
        body = await request.text()
        if not body or not body.strip():
            return web.Response(status=400, text="Empty request body")
        data = json.loads(body)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return web.Response(status=200, text="Saved")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving {section} presets: {e}")
        return web.Response(status=500, text=str(e))

@server.PromptServer.instance.routes.post("/configbuilder/load_config")
async def load_config(request):
    try:
        data = await request.json()
        filename = data.get("name")
        
        if not filename:
             return web.Response(status=400, text="Missing name")
             
        if not filename.endswith(".json"):
            filename += ".json"

        filename = os.path.basename(filename)
        filepath = os.path.join(CONFIGS_DIR, filename)

        if not _is_path_within(filepath, CONFIGS_DIR):
            return web.Response(status=403, text="Forbidden")

        if not os.path.exists(filepath):
            return web.Response(status=404, text="Config not found")
            
        with open(filepath, "r") as f:
            config_data = json.load(f)
            
        return web.json_response(config_data)
    except Exception as e:
        print(f"[ConfigBuilder] Error loading config: {e}")
        return web.Response(status=500, text=str(e))


# =============================================================================
# API: DELETE SESSION
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/delete_session")
async def delete_session(request):
    try:
        data = await request.json()
        session_name = data.get("session_name")

        # Sanitize
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)

        if not session_name or session_name == "default_session":
             return web.Response(status=400, text="Invalid session name")

        # Path construction with containment check
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            return web.Response(status=200, text="Deleted")
        else:
            return web.Response(status=404, text="Session not found")

    except Exception as e:
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: LIST SESSIONS
# =============================================================================

@server.PromptServer.instance.routes.get("/config_tester/list_sessions")
async def list_sessions(request):
    """
    List all available sessions with metadata for the session picker.
    Returns JSON array sorted by modification time (newest first).
    Each entry: {name, item_count, first_image, mtime}
    """
    try:
        benchmarks_dir = _get_benchmarks_base()
        sessions = []

        if os.path.exists(benchmarks_dir):
            for item in os.listdir(benchmarks_dir):
                item_path = os.path.join(benchmarks_dir, item)
                manifest_path = os.path.join(item_path, "manifest.json")

                if os.path.isdir(item_path) and os.path.exists(manifest_path):
                    try:
                        mtime = os.path.getmtime(item_path)
                        # Read manifest for item count and first image
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        items = manifest.get("items", [])
                        item_count = len(items)

                        # Find first image filename for thumbnail
                        first_image = None
                        if items:
                            first_item = items[0]
                            fname = first_item.get("filename")
                            if fname:
                                first_image = f"benchmarks/{item}/{fname}"

                        sessions.append({
                            "name": item,
                            "item_count": item_count,
                            "first_image": first_image,
                            "mtime": mtime
                        })
                    except Exception:
                        # Skip sessions with unreadable manifests
                        sessions.append({
                            "name": item,
                            "item_count": 0,
                            "first_image": None,
                            "mtime": os.path.getmtime(item_path)
                        })

        # Sort by modification time, newest first
        sessions.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response(sessions)

    except Exception as e:
        return web.json_response([], status=500)


# =============================================================================
# API: SAVE CHANGES (Optimized - Only Changed Items)
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/save_changes")
async def save_changes(request):
    """
    OPTIMIZED: Save only changed items instead of full manifest.
    
    This drastically reduces network payload and processing time:
    - Before: ~10MB for 500 images
    - After: ~10KB for 1-5 changed items
    
    Receives:
        - session_name: str
        - changed_items: list of item objects with updates
    
    Process:
        1. Load current manifest from disk
        2. Update only the changed items by ID
        3. Save back to disk
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")
        changed_items = data.get("changed_items", [])
        
        # Sanitize
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)
        
        if not session_name or not changed_items:
            return web.Response(status=400, text="Missing session_name or changed_items")
        
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")

        # Load current manifest
        if not os.path.exists(manifest_path):
            return web.Response(status=404, text=f"Session '{session_name}' not found")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Create lookup of changed items by ID
        changed_by_id = {item.get("id"): item for item in changed_items if "id" in item}
        
        # Update items in manifest
        items_updated = 0
        for i, item in enumerate(manifest.get("items", [])):
            item_id = item.get("id")
            if item_id in changed_by_id:
                # Merge changes (preserve fields not sent by client)
                updated_item = changed_by_id[item_id]
                manifest["items"][i].update(updated_item)
                items_updated += 1
        
        # Save updated manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        
        print(f"[ConfigTester] ⚡ Updated {items_updated} items in {session_name}")
        return web.Response(status=200, text=f"Updated {items_updated} items")
        
    except Exception as e:
        print(f"[ConfigTester] Error saving changes: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))
    
# =============================================================================
# API: PAIRWISE REVIEW VOTES (manifest "comparisons")
# =============================================================================

def _comparisons_manifest_path(session_name):
    """Resolve a sanitized session name to its manifest path; returns (path, error_response)."""
    if session_name:
        session_name = re.sub(r'[^\w\-]', '', str(session_name))
    if not session_name:
        return None, web.Response(status=400, text="Missing session_name")
    base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)
    if not _is_path_within(base_dir, _get_benchmarks_base()):
        return None, web.Response(status=403, text="Forbidden: path outside benchmarks directory")
    manifest_path = os.path.join(base_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None, web.Response(status=404, text=f"Session '{session_name}' not found")
    return manifest_path, None


@server.PromptServer.instance.routes.post("/config_tester/save_comparisons")
async def save_comparisons(request):
    """
    Append / remove blind-compare votes in manifest["comparisons"].
    Body: {session_name, append: [{a,b,winner,ts,mode,blind?}], remove: ["ts:a:b", ...]}
    Only the comparisons list is touched; items/meta are written back as read.
    """
    try:
        from .manifest_utils import update_manifest_comparisons
        data = await request.json()
        append = data.get("append") or []
        remove = data.get("remove") or []
        if not isinstance(append, list) or not isinstance(remove, list):
            return web.Response(status=400, text="append and remove must be lists")
        if (not append and not remove) or len(append) > 5000 or len(remove) > 5000:
            return web.Response(status=400, text="Nothing to save or too many votes")
        manifest_path, err = _comparisons_manifest_path(data.get("session_name"))
        if err:
            return err
        try:
            result = await asyncio.to_thread(
                update_manifest_comparisons, manifest_path, append,
                [str(k) for k in remove])
        except FileNotFoundError:
            return web.Response(status=404, text="Session manifest not found")
        except ValueError:  # includes json.JSONDecodeError (manifest mid-write): client retries
            return web.Response(status=409, text="Manifest unreadable, retry")
        print(f"[ConfigTester] ⚖️ Votes: +{result['added']} -{result['removed']} = {result['total']} in {data.get('session_name')}")
        return web.json_response(result)
    except Exception as e:
        print(f"[ConfigTester] Error saving comparisons: {e}")
        return web.Response(status=500, text=str(e))


@server.PromptServer.instance.routes.post("/config_tester/clear_comparisons")
async def clear_comparisons(request):
    """Delete every vote of a session (body: {session_name})."""
    try:
        from .manifest_utils import update_manifest_comparisons
        data = await request.json()
        manifest_path, err = _comparisons_manifest_path(data.get("session_name"))
        if err:
            return err
        try:
            result = await asyncio.to_thread(update_manifest_comparisons, manifest_path, (), (), True)
        except FileNotFoundError:
            return web.Response(status=404, text="Session manifest not found")
        except ValueError:
            return web.Response(status=409, text="Manifest unreadable, retry")
        print(f"[ConfigTester] ⚖️ Votes cleared ({result['removed']}) in {data.get('session_name')}")
        return web.json_response(result)
    except Exception as e:
        print(f"[ConfigTester] Error clearing comparisons: {e}")
        return web.Response(status=500, text=str(e))


# =============================================================================
# API: SAVE MANIFEST (Legacy - Full Save)
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/save_manifest")
async def save_manifest(request):
    """
    Save manifest from dashboard.
    
    CRITICAL: This is called when users favorite/reject/note images in the dashboard.
    We need to preserve any NEW images that generation added but dashboard doesn't know about yet.
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")
        manifest_data = data.get("manifest")  # This is from dashboard (may be stale)
        
        # --- sanitize ---
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)

        if not session_name or not manifest_data:
            return web.Response(status=400, text="Missing session_name or manifest")

        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")

        # --- MERGE STRATEGY: Preserve server data ---
        # 1. Load server manifest (has newest images)
        server_manifest = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    server_manifest = json.load(f)
            except:
                pass

        if server_manifest:
            # Build lookup of items by ID from dashboard
            dashboard_items = {item.get("id"): item for item in manifest_data.get("items", [])}
            
            # Merge: Update existing items, keep new items
            merged_items = []
            for server_item in server_manifest.get("items", []):
                item_id = server_item.get("id")
                if item_id in dashboard_items:
                    # Item exists in dashboard: merge updates
                    dashboard_item = dashboard_items[item_id]
                    # Preserve server's metadata but update user actions
                    merged_item = server_item.copy()
                    merged_item["favorited"] = dashboard_item.get("favorited", False)
                    merged_item["rejected"] = dashboard_item.get("rejected", False)
                    merged_item["note"] = dashboard_item.get("note", "")
                    merged_items.append(merged_item)
                else:
                    # NEW item from server (generation added it): keep as-is
                    merged_items.append(server_item)
            
            # Update manifest
            manifest_data["items"] = merged_items
            
            # Preserve server's meta
            if "meta" in server_manifest:
                manifest_data["meta"] = server_manifest["meta"]

            # Preserve pairwise-review votes (disk wins; the dashboard copy can be stale)
            if "comparisons" in server_manifest:
                manifest_data["comparisons"] = server_manifest["comparisons"]

        # Save merged manifest
        os.makedirs(base_dir, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)

        return web.Response(status=200, text="Saved")

    except Exception as e:
        print(f"[ConfigTester] Error saving manifest: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: GET SESSION HTML
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/get_session_html")
async def get_session_html(request):
    """
    Dynamically generate HTML for a session.
    This allows the dashboard to load without triggering a workflow execution.
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")
        node_id = data.get("node_id")

        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)

        if not session_name:
            # No session name — return landing page with empty manifest
            empty_manifest = {"items": [], "meta": {"model": "", "positive": "", "negative": ""}, "session_name": ""}
            html = get_html_template("", empty_manifest, node_id)
            return web.Response(status=200, text=html)

        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            # No session found — return full template with empty manifest
            # so the landing page JS can show available sessions
            empty_manifest = {"items": [], "meta": {"model": "", "positive": "", "negative": ""}, "session_name": session_name}
            html = get_html_template(session_name, empty_manifest, node_id)
            return web.Response(status=200, text=html)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Generate HTML on the fly
        html = get_html_template(session_name, manifest, node_id)
        return web.Response(status=200, text=html)

    except Exception as e:
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: STANDALONE DASHBOARD PAGE + HEADLESS RUN API
# =============================================================================
_run_tracker = RunTracker()

@server.PromptServer.instance.routes.get("/config_tester/dashboard")
async def standalone_dashboard(request):
    """
    Serve the dashboard SPA as a top-level page (phone / second-monitor friendly).
    Optional ?session=<name>; without it the SPA shows its session landing cards.
    """
    try:
        session_name = sanitize_session_name_strict(request.query.get("session", ""))
        headers = {"Cache-Control": "no-store"}
        empty_manifest = {"items": [], "meta": {"model": "", "positive": "", "negative": ""}, "session_name": ""}

        if not session_name:
            html = get_html_template("", empty_manifest, "standalone", standalone=True)
            return web.Response(status=200, text=html, content_type="text/html", charset="utf-8", headers=headers)

        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)
        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            # Unknown session: landing page (picker); the title stays empty so the SPA does not
            # pretend a session is loaded.
            html = get_html_template("", empty_manifest, "standalone", standalone=True)
            return web.Response(status=200, text=html, content_type="text/html", charset="utf-8", headers=headers)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        html = get_html_template(session_name, manifest, "standalone", standalone=True)
        return web.Response(status=200, text=html, content_type="text/html", charset="utf-8", headers=headers)
    except Exception as e:
        print(f"[GridTester] ❌ /dashboard failed: {e}")
        return web.Response(status=500, text=str(e))


@server.PromptServer.instance.routes.post("/config_tester/run")
async def run_headless(request):
    """
    Queue a USCG run without the ComfyUI graph editor (Discord bot / session hub / curl).
    Builds the 2-node prompt (Generator -> Dashboard) and enqueues it the way
    PromptServer's own POST /prompt handler does (in-process, no HTTP self-call).
    Body: {configs_json?: object|string|list, config_name?: str, session_name?: str,
           overwrite_existing?: bool, force?: bool (default true), front?: bool, client_id?: str}
    force=false lets ComfyUI's node cache short-circuit an identical request.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Request body must be JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "Request body must be a JSON object"}, status=400)

        has_inline = body.get("configs_json") not in (None, "")
        has_named = bool(body.get("config_name"))
        if has_inline == has_named:
            return web.json_response({"error": "Provide exactly one of 'configs_json' or 'config_name'"}, status=400)

        try:
            if has_named:
                try:
                    cfg_path = resolve_saved_config(CONFIGS_DIR, body["config_name"])
                except FileNotFoundError:
                    return web.json_response({"error": f"Config '{body['config_name']}' not found"}, status=404)
                except PermissionError:
                    return web.json_response({"error": "Forbidden"}, status=403)
                with open(cfg_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                obj = normalize_configs_payload(
                    saved_config_to_configs_obj(saved, UltimateConfigBuilder.state_to_configs_json))
            else:
                obj = normalize_configs_payload(body["configs_json"])
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        force = body.get("force", True)
        force = True if force is None else bool(force)
        overwrite = body.get("overwrite_existing")
        obj, session_name = apply_run_overrides(
            obj,
            session_name=body.get("session_name") or None,
            overwrite_existing=overwrite if overwrite is None else bool(overwrite),
            nonce=uuid.uuid4().hex if force else None,
        )
        if not session_name:
            return web.json_response({"error": "session_name is empty after sanitising (allowed: letters, digits, _ and -)"}, status=400)

        configs_str = json.dumps(obj, indent=2, ensure_ascii=False)
        prompt = build_run_prompt(configs_str, session_name)

        client_id = body.get("client_id")
        client_id = str(client_id) if client_id else None
        front = bool(body.get("front", False))

        import execution  # ComfyUI core; imported lazily so this package still loads without it
        # --- enqueue: mirrors PromptServer.post_prompt (ComfyUI 0.42.5, server.py) ---
        ps = server.PromptServer.instance
        json_data = {"prompt": prompt}
        if client_id:
            json_data["client_id"] = client_id
        if front:
            json_data["front"] = True
        json_data = ps.trigger_on_prompt(json_data)  # same hook third-party on_prompt handlers use
        if not isinstance(json_data, dict) or "prompt" not in json_data:
            return web.json_response({"error": "An on_prompt handler rejected the run"}, status=400)
        prompt = json_data["prompt"]
        if "number" in json_data:
            number = float(json_data["number"])
        else:
            number = ps.number
            if json_data.get("front"):
                number = -number
            ps.number += 1

        prompt_id = str(uuid.uuid4())
        if hasattr(ps, "node_replace_manager"):
            ps.node_replace_manager.apply_replacements(prompt)

        # 0.42.5: async validate_prompt(prompt_id, prompt, partial_execution_list). Older builds
        # take (prompt) or (prompt_id, prompt) and may be sync.
        n_params = len(inspect.signature(execution.validate_prompt).parameters)
        if n_params >= 3:
            valid = execution.validate_prompt(prompt_id, prompt, None)
        elif n_params == 2:
            valid = execution.validate_prompt(prompt_id, prompt)
        else:
            valid = execution.validate_prompt(prompt)
        if inspect.isawaitable(valid):
            valid = await valid
        if not valid[0]:
            print(f"[GridTester] ⚠️ /run rejected by ComfyUI validation: {valid[1]}")
            return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)

        extra_data = {}
        if client_id:
            extra_data["client_id"] = client_id
        extra_data["create_time"] = int(time.time() * 1000)  # milliseconds, like post_prompt
        item = (number, prompt_id, prompt, extra_data, valid[2])
        if hasattr(execution, "SENSITIVE_EXTRA_DATA_KEYS"):
            item = item + ({},)  # trailing `sensitive` dict (empty: nothing sensitive to strip)
        ps.prompt_queue.put(item)

        _run_tracker.add(prompt_id, session_name)
        dashboard_path = "/config_tester/dashboard?session=" + urllib.parse.quote(session_name)
        print(f"[GridTester] 🚀 /run queued {prompt_id} session={session_name}")
        return web.json_response({
            "prompt_id": prompt_id,
            "number": number,
            "session_name": session_name,
            "dashboard_path": dashboard_path,
            "dashboard_url": f"{request.scheme}://{request.host}{dashboard_path}",
            "node_errors": valid[3],
        })
    except Exception as e:
        print(f"[GridTester] ❌ /run failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/config_tester/run_status")
async def run_status(request):
    """State of a run queued through /config_tester/run: pending | running | success | error | unknown."""
    try:
        prompt_id = request.query.get("prompt_id", "")
        if not prompt_id:
            return web.json_response({"error": "Missing prompt_id"}, status=400)
        pq = server.PromptServer.instance.prompt_queue
        getter = getattr(pq, "get_current_queue_volatile", None) or pq.get_current_queue
        running, pending = getter()
        result = {"prompt_id": prompt_id, "queue_remaining": len(running) + len(pending)}

        hist = pq.get_history(prompt_id=prompt_id).get(prompt_id)
        found = find_in_queue(running, pending, prompt_id)
        if found:
            result["state"] = found[0]
            if found[0] == "pending":
                result["queue_position"] = found[1]
        elif hist is not None:
            summary = summarize_history(hist)
            result["status"] = summary
            result["state"] = "error" if summary["status"] == "error" else "success"
        else:
            result["state"] = "unknown"
            return web.json_response(result, status=404)

        session_name = _run_tracker.get(prompt_id)
        if session_name:
            dashboard_path = "/config_tester/dashboard?session=" + urllib.parse.quote(session_name)
            result["session_name"] = session_name
            result["dashboard_url"] = f"{request.scheme}://{request.host}{dashboard_path}"
            manifest_path = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name, "manifest.json")
            if _is_path_within(manifest_path, _get_benchmarks_base()) and os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        result["item_count"] = len(json.load(f).get("items", []))
                except Exception:
                    pass
        return web.json_response(result)
    except Exception as e:
        print(f"[GridTester] ❌ /run_status failed: {e}")
        return web.json_response({"error": str(e)}, status=500)

# =============================================================================
# API: EXPORT FAVORITES
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/export_favorites")
async def export_favorites(request):
    """
    Export favorited images to a 'benchmark_favorites/{session_name}' folder.
    Optionally packs metadata into images for CivitAI uploads.
    Optionally organizes into subfolders by unique prompts.
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")
        pack_metadata = data.get("pack_metadata", False)
        organize_by_prompt = data.get("organize_by_prompt", False)
        organize_by_lora = data.get("organize_by_lora", False)
        export_prompt_txt = data.get("export_prompt_txt", False)
        copy_manifest = data.get("copy_manifest", True)
        pack_workflow = data.get("pack_workflow", False)
        pack_nodes_workflow = data.get("pack_nodes_workflow", False)
        workflow_data = data.get("workflow_data", None)
        nodes_workflows = data.get("nodes_workflows", None)

        # Sanitize
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)
        
        if not session_name:
            return web.Response(status=400, text="Missing session_name")

        # Paths
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")
        images_dir = os.path.join(base_dir, "images")

        # Load manifest
        if not os.path.exists(manifest_path):
            return web.Response(status=404, text=f"Session '{session_name}' not found")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Filter favorited items
        favorited = [item for item in manifest.get("items", []) if item.get("favorited", False)]
        
        if not favorited:
            return web.Response(status=200, text="No favorited images to export")
        
        # Build prompt mapping if organizing by prompt
        prompt_to_folder = {}
        if organize_by_prompt:
            unique_prompts = []
            for item in favorited:
                prompt = item.get("positive") or manifest.get("meta", {}).get("positive", "")
                if prompt and prompt not in unique_prompts:
                    unique_prompts.append(prompt)
            
            # Create Prompt1, Prompt2, etc. mapping
            for idx, prompt in enumerate(unique_prompts, 1):
                prompt_to_folder[prompt] = f"Prompt{idx}"
            
            print(f"[Export] Organizing into {len(unique_prompts)} prompt folders")
        
        # Build lora mapping if organizing by lora
        lora_to_folder = {}
        if organize_by_lora:
            unique_loras = []
            for item in favorited:
                lora = item.get("lora_expanded") or manifest.get("meta", {}).get("lora", "")
                if lora and lora not in unique_loras:
                    unique_loras.append(lora)
            
            # Create Lora1, Lora2, etc. mapping
            for idx, lora in enumerate(unique_loras, 1):
                lora_to_folder[lora] = f"Loras{idx}"
            
            print(f"[Export] Organizing into {len(unique_loras)} lora folders")
        
        # Create base export directory
        export_base = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name, "favorites")
        os.makedirs(export_base, exist_ok=True)
        
        exported_count = 0
        
        for item in favorited:
            # Get source image path - handle both URL format and relative paths
            file_path = item.get("file", "")
            
            # Parse filename from URL format: /view?filename=img_123.webp&type=output&subfolder=benchmarks/Session/images
            if file_path.startswith("/view?"):
                # Extract filename and subfolder from URL
                parsed_url = urllib.parse.urlparse(file_path)
                url_params = urllib.parse.parse_qs(parsed_url.query)
                filename = url_params.get("filename", [""])[0]
                subfolder = url_params.get("subfolder", [""])[0]
                if subfolder:
                    source_path = os.path.join(folder_paths.get_output_directory(), subfolder, filename)
                else:
                    source_path = os.path.join(images_dir, filename)
            elif file_path.startswith("./images/"):
                # Relative path format
                filename = file_path[9:]  # Remove ./images/
                source_path = os.path.join(images_dir, filename)
            else:
                # Just use basename
                filename = os.path.basename(file_path)
                source_path = os.path.join(images_dir, filename)

            if not filename:
                print(f"[Export] Warning: Could not parse filename from: {file_path}")
                continue
            
            if not os.path.exists(source_path):
                print(f"[Export] Warning: Image not found: {source_path}")
                continue
            
            # Determine destination folder
            if organize_by_prompt and organize_by_lora:
                # Both options: create nested folders Prompt/Lora
                prompt = item.get("positive") or manifest.get("meta", {}).get("positive", "")
                lora = item.get("lora") or manifest.get("meta", {}).get("lora", "")
                prompt_folder = prompt_to_folder.get(prompt, "Unknown")
                lora_folder = lora_to_folder.get(lora, "Unknown")
                dest_dir = os.path.join(export_base, prompt_folder, lora_folder)
                os.makedirs(dest_dir, exist_ok=True)
            elif organize_by_prompt:
                prompt = item.get("positive") or manifest.get("meta", {}).get("positive", "")
                folder_name = prompt_to_folder.get(prompt, "Unknown")
                dest_dir = os.path.join(export_base, folder_name)
                os.makedirs(dest_dir, exist_ok=True)
            elif organize_by_lora:
                lora = item.get("lora") or manifest.get("meta", {}).get("lora", "")
                folder_name = lora_to_folder.get(lora, "Unknown")
                dest_dir = os.path.join(export_base, folder_name)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = export_base
            
            # Destination path
            dest_filename = filename
            if pack_metadata:
                # Change extension to .png if packing metadata
                dest_filename = os.path.splitext(filename)[0] + '.png'
            
            dest_path = os.path.join(dest_dir, dest_filename)
            
            # Copy or pack metadata
            if pack_metadata:
                try:
                    # Determine which workflow to embed
                    item_workflow = None
                    if pack_nodes_workflow and nodes_workflows:
                        # Per-image nodes workflow takes priority
                        item_workflow = nodes_workflows.get(item.get("file", ""))
                    elif pack_workflow and workflow_data:
                        # Full ComfyUI graph workflow
                        item_workflow = workflow_data
                    pack_metadata_into_image(source_path, dest_path, item, manifest.get("meta", {}), workflow_data=item_workflow)
                    exported_count += 1
                except Exception as e:
                    print(f"[Export] Error packing metadata for {filename}: {e}")
                    # Fallback to simple copy
                    shutil.copy2(source_path, dest_path)
                    exported_count += 1
            else:
                shutil.copy2(source_path, dest_path)
                exported_count += 1

            # Export positive prompt as .txt file alongside the image
            if export_prompt_txt:
                try:
                    prompt_text = item.get("positive") or manifest.get("meta", {}).get("positive", "")
                    if prompt_text:
                        txt_filename = os.path.splitext(dest_filename)[0] + '.txt'
                        txt_path = os.path.join(dest_dir, txt_filename)
                        with open(txt_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(prompt_text)
                except Exception as e:
                    print(f"[Export] Error writing prompt txt for {filename}: {e}")
        
        # Copy cleaned favorites-only manifest if requested
        if copy_manifest:
            try:
                cleaned_manifest = {
                    "items": favorited,
                    "meta": manifest.get("meta", {}),
                    "session_name": session_name
                }
                manifest_dest = os.path.join(export_base, "manifest.json")
                with open(manifest_dest, "w", encoding="utf-8") as f:
                    json.dump(cleaned_manifest, f, indent=2, ensure_ascii=False)
                print(f"[Export] Saved cleaned favorites manifest to {manifest_dest}")
            except Exception as e:
                print(f"[Export] Error saving cleaned manifest: {e}")

        result_msg = f"Exported {exported_count} favorited images to 'benchmarks/{session_name}/favorites/'"
        if organize_by_prompt and organize_by_lora:
            result_msg += f" (organized into {len(prompt_to_folder)} prompt folders with {len(lora_to_folder)} lora subfolders)"
        elif organize_by_prompt:
            result_msg += f" (organized into {len(prompt_to_folder)} prompt folders)"
        elif organize_by_lora:
            result_msg += f" (organized into {len(lora_to_folder)} lora folders)"
        if pack_metadata:
            result_msg += " (with metadata packed)"
        if export_prompt_txt:
            result_msg += " (with prompt .txt files)"
        if copy_manifest:
            result_msg += " (with favorites manifest)"
        if pack_workflow:
            result_msg += " (with full workflow)"
        if pack_nodes_workflow:
            result_msg += " (with nodes workflows)"

        print(f"[ConfigTester] ✅ {result_msg}")
        return web.Response(status=200, text=result_msg)
        
    except Exception as e:
        print(f"[ConfigTester] Error exporting favorites: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: DELETE NON-FAVORITED ITEMS
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/delete_non_favorites")
async def delete_non_favorites(request):
    """
    Delete all non-favorited images from a session and update the manifest.
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")

        # Sanitize
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)

        if not session_name:
            return web.Response(status=400, text="Missing session_name")

        # Paths
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")
        images_dir = os.path.join(base_dir, "images")

        # Load manifest
        if not os.path.exists(manifest_path):
            return web.Response(status=404, text=f"Session '{session_name}' not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        items = manifest.get("items", [])
        favorited = [item for item in items if item.get("favorited", False)]
        non_favorited = [item for item in items if not item.get("favorited", False)]

        if not non_favorited:
            return web.Response(status=200, text="No non-favorited items to delete")

        # Delete non-favorited image files
        deleted_count = 0
        for item in non_favorited:
            file_path = item.get("file", "")

            # Parse filename from various formats
            if file_path.startswith("/view?"):
                parsed_url = urllib.parse.urlparse(file_path)
                url_params = urllib.parse.parse_qs(parsed_url.query)
                filename = url_params.get("filename", [""])[0]
            elif file_path.startswith("./images/"):
                filename = file_path[9:]
            elif "filename=" in file_path:
                filename = file_path.split("filename=")[-1].split("&")[0]
            else:
                filename = os.path.basename(file_path)

            if filename:
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"[Delete] Error deleting {filename}: {e}")

        # Update manifest to only contain favorited items
        manifest["items"] = favorited
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        result_msg = f"Deleted {deleted_count} non-favorited images. {len(favorited)} favorited items remain."
        print(f"[ConfigTester] 🗑️ {result_msg}")
        return web.Response(status=200, text=result_msg)

    except Exception as e:
        print(f"[ConfigTester] Error deleting non-favorites: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: DELETE REJECTED ITEMS
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/delete_rejected")
async def delete_rejected(request):
    """
    Delete all rejected images from a session and update the manifest.
    """
    try:
        data = await request.json()
        session_name = data.get("session_name")

        # Sanitize
        if session_name:
            session_name = re.sub(r'[^\w\-]', '', session_name)

        if not session_name:
            return web.Response(status=400, text="Missing session_name")

        # Paths
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.Response(status=403, text="Forbidden: path outside benchmarks directory")

        manifest_path = os.path.join(base_dir, "manifest.json")
        images_dir = os.path.join(base_dir, "images")

        # Load manifest
        if not os.path.exists(manifest_path):
            return web.Response(status=404, text=f"Session '{session_name}' not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        items = manifest.get("items", [])
        rejected = [item for item in items if item.get("rejected", False)]
        kept = [item for item in items if not item.get("rejected", False)]

        if not rejected:
            return web.Response(status=200, text="No rejected items to delete")

        # Delete rejected image files
        deleted_count = 0
        for item in rejected:
            file_path = item.get("file", "")

            # Parse filename from various formats
            if file_path.startswith("/view?"):
                parsed_url = urllib.parse.urlparse(file_path)
                url_params = urllib.parse.parse_qs(parsed_url.query)
                filename = url_params.get("filename", [""])[0]
            elif file_path.startswith("./images/"):
                filename = file_path[9:]
            elif "filename=" in file_path:
                filename = file_path.split("filename=")[-1].split("&")[0]
            else:
                filename = os.path.basename(file_path)

            if filename:
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"[Delete] Error deleting {filename}: {e}")

        # Tombstone rejected PREVIEW cells: a later full pass must still know the user rejected them
        # even though the preview item (the only record of the rejection) is gone after this delete.
        from .config_utils import collect_rejected_preview_fps
        manifest.setdefault("meta", {})["rejected_preview_fps"] = collect_rejected_preview_fps(
            rejected, manifest.get("meta", {}).get("rejected_preview_fps", []))

        # Update manifest to remove rejected items
        manifest["items"] = kept
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        result_msg = f"Deleted {deleted_count} rejected images. {len(kept)} items remain."
        print(f"[ConfigTester] 🗑️ {result_msg}")
        return web.Response(status=200, text=result_msg)

    except Exception as e:
        print(f"[ConfigTester] Error deleting rejected: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

# =============================================================================
# API: SCAN EXTERNAL DIRECTORY
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/scan_directory")
async def scan_directory_route(request):
    """
    Scan an external directory for images, extract PNG metadata,
    and generate a manifest compatible with the dashboard.
    Creates symlinks in the output directory so ComfyUI's built-in
    /view endpoint can serve the images (no custom file serving needed).
    """
    try:
        data = await request.json()
        directory_path = data.get("directory_path", "").strip()
        session_name = data.get("session_name", "").strip()

        if not directory_path:
            return web.json_response({"error": "Missing directory_path"}, status=400)

        # Normalize path
        directory_path = os.path.normpath(directory_path)

        if not os.path.isabs(directory_path):
            return web.json_response({"error": "Path must be absolute"}, status=400)

        if not os.path.isdir(directory_path):
            return web.json_response({"error": f"Directory not found: {directory_path}"}, status=404)

        # Auto-generate session name from directory if not provided
        if not session_name:
            dir_basename = os.path.basename(directory_path)
            session_name = "scan-" + re.sub(r'[^\w\-]', '', dir_basename)

        # Sanitize session name
        session_name = re.sub(r'[^\w\-]', '', session_name)
        if not session_name:
            session_name = "scan-unnamed"

        print(f"[DirScanner] Scanning directory: {directory_path}")
        print(f"[DirScanner] Session name: {session_name}")

        # Prepare the symlink directory within the output folder
        # so ComfyUI's built-in /view endpoint can serve the images
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)

        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.json_response({"error": "Forbidden: path outside benchmarks directory"}, status=403)

        link_dir = os.path.join(base_dir, "external_images")
        os.makedirs(link_dir, exist_ok=True)

        # The subfolder path for /view URLs (relative to output dir)
        view_subfolder = f"benchmarks/{session_name}/external_images"

        # Run the scan in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        items, stats = await loop.run_in_executor(
            None, scan_directory_for_images, directory_path, 5000, view_subfolder
        )

        from_manifest = stats.get("from_manifest", 0)
        manifest_info = f", {from_manifest} from existing manifest" if from_manifest > 0 else ""
        print(f"[DirScanner] Found {stats['total']} images ({stats['with_metadata']} with metadata, {stats['skipped']} skipped{manifest_info})")

        # Create symlinks (or copy on Windows if symlinks not available)
        # for each image so /view can serve them
        _create_image_links(directory_path, link_dir, items)

        # Build manifest
        manifest = {
            "session_name": session_name,
            "items": items,
            "meta": {
                "source_directory": directory_path,
                "scan_type": "directory",
                "total_scanned": stats["total"],
                "with_metadata": stats["with_metadata"],
            }
        }

        # Save manifest to benchmarks session directory
        manifest_path = os.path.join(base_dir, "manifest.json")

        # Re-scan of an existing session: keep ids, favorites/rejects/notes, metrics and pairwise votes
        # (merged under the manifest lock, written atomically)
        from .manifest_utils import write_rescan_manifest
        write_rescan_manifest(manifest_path, manifest)

        print(f"[DirScanner] Created image links in: {link_dir}")

        from_manifest = stats.get("from_manifest", 0)
        return web.json_response({
            "session_name": session_name,
            "item_count": len(items),
            "with_metadata": stats["with_metadata"],
            "without_metadata": len(items) - stats["with_metadata"],
            "skipped": stats["skipped"],
            "from_manifest": from_manifest,
        })

    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        print(f"[DirScanner] Error: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# API: DASHBOARD UPSCALE
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/upscale_images")
async def upscale_images(request):
    """Start an async upscale job from the dashboard."""
    try:
        data = await request.json()
        session_name = data.get("session_name", "")
        image_ids = data.get("image_ids", [])
        upscale_config = data.get("upscale_config", {})
        all_favorited = data.get("all_favorited", False)

        if not session_name:
            return web.json_response({"error": "Missing session_name"}, status=400)
        if not upscale_config or not upscale_config.get("pipelines"):
            return web.json_response({"error": "Missing upscale_config with pipelines"}, status=400)

        from .upscale_runner import start_upscale_job
        job_id, error = start_upscale_job(session_name, image_ids, upscale_config, all_favorited=all_favorited)

        if error:
            return web.json_response({"error": error}, status=400)

        from .upscale_runner import get_upscale_status
        status = get_upscale_status(job_id)
        return web.json_response({"job_id": job_id, "total_images": status["total"]})

    except Exception as e:
        print(f"[ConfigTester] Error starting upscale job: {e}")
        return web.json_response({"error": str(e)}, status=500)

@server.PromptServer.instance.routes.get("/config_tester/upscale_status")
async def upscale_status(request):
    """Check status of an async upscale job."""
    job_id = request.query.get("job_id", "")
    if not job_id:
        return web.json_response({"error": "Missing job_id"}, status=400)
    from .upscale_runner import get_upscale_status
    return web.json_response(get_upscale_status(job_id))

@server.PromptServer.instance.routes.post("/config_tester/cancel_upscale")
async def cancel_upscale(request):
    """Cancel a running upscale job."""
    try:
        data = await request.json()
        job_id = data.get("job_id", "")
        from .upscale_runner import cancel_upscale_job
        cancelled = cancel_upscale_job(job_id)
        return web.json_response({"cancelled": cancelled})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# --- Image metrics for an EXISTING session (background job, CPU only) ---
@server.PromptServer.instance.routes.post("/config_tester/compute_metrics")
async def compute_metrics(request):
    """Start a job that computes image metrics for the session's items that have none."""
    try:
        data = await request.json()
        session_name = re.sub(r'[^\w\-]', '', str(data.get("session_name") or ""))
        if not session_name:
            return web.json_response({"error": "Missing session_name"}, status=400)
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)
        if not _is_path_within(base_dir, _get_benchmarks_base()):
            return web.json_response({"error": "Forbidden: path outside benchmarks directory"}, status=403)
        from .metrics_runner import start_metrics_job, get_metrics_status
        job_id, error = start_metrics_job(session_name, overwrite=bool(data.get("overwrite", False)))
        if error:
            return web.json_response({"error": error}, status=400)
        return web.json_response({"job_id": job_id, "total_images": get_metrics_status(job_id)["total"]})
    except Exception as e:
        print(f"[ConfigTester] Error starting metrics job: {e}")
        return web.json_response({"error": str(e)}, status=500)

@server.PromptServer.instance.routes.get("/config_tester/metrics_status")
async def metrics_status(request):
    job_id = request.query.get("job_id", "")
    if not job_id:
        return web.json_response({"error": "Missing job_id"}, status=400)
    from .metrics_runner import get_metrics_status
    return web.json_response(get_metrics_status(job_id))

@server.PromptServer.instance.routes.post("/config_tester/cancel_metrics")
async def cancel_metrics(request):
    try:
        data = await request.json()
        from .metrics_runner import cancel_metrics_job
        return web.json_response({"cancelled": cancel_metrics_job(data.get("job_id", ""))})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# API: WORKFLOW SWEEP (sweep any API-format workflow through ComfyUI's own executor)
# JSON bodies only, no path parameters: workflow JSON is request-body data, never a filesystem path.
# =============================================================================

@server.PromptServer.instance.routes.post("/config_tester/sweep/parse")
async def sweep_parse(request):
    """Describe an API-format workflow for the Builder (nodes, outputs, image sources, suggested bindings)."""
    try:
        data = await request.json()
        from .workflow_sweep import parse_for_ui
        return web.json_response(parse_for_ui(data.get("api_json")))
    except ValueError as e:
        return web.json_response({"ok": False, "errors": [str(e)], "error": str(e)}, status=400)
    except Exception as e:
        print(f"[ConfigTester] Error parsing workflow: {e}")
        return web.json_response({"ok": False, "errors": [str(e)], "error": str(e)}, status=500)

@server.PromptServer.instance.routes.post("/config_tester/sweep/dry_run")
async def sweep_dry_run(request):
    """First N concrete prompts (with collector) + validation of job 1. Nothing is queued."""
    try:
        data = await request.json()
        from .workflow_sweep_runner import dry_run
        return web.json_response(await dry_run(data))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        print(f"[ConfigTester] Error in sweep dry run: {e}")
        return web.json_response({"error": str(e)}, status=500)

@server.PromptServer.instance.routes.post("/config_tester/sweep/start")
async def sweep_start(request):
    """Start a workflow sweep from the Builder (no Generator node needed; also the resume path)."""
    try:
        data = await request.json()
        from .workflow_sweep_runner import api_start
        status, body = await api_start(data)
        return web.json_response(body, status=status)
    except Exception as e:
        print(f"[ConfigTester] Error starting sweep: {e}")
        return web.json_response({"error": str(e)}, status=500)

@server.PromptServer.instance.routes.get("/config_tester/sweep/status")
async def sweep_status(request):
    from .workflow_sweep_runner import get_status
    status = get_status(request.query.get("sweep_id") or None, request.query.get("session_name") or None)
    if status is None:
        return web.json_response({"error": "No such sweep"}, status=404)
    return web.json_response(status)

@server.PromptServer.instance.routes.get("/config_tester/sweep/job_prompt")
async def sweep_job_prompt(request):
    """The API-format prompt of one sweep job (derived JSON, no file is served; ids only)."""
    try:
        from .workflow_sweep_runner import job_prompt
        status, body = job_prompt(request.query.get("session_name") or "", request.query.get("sweep_id") or "",
                                  request.query.get("job_index", ""))
        return web.json_response(body, status=status)
    except Exception as e:
        print(f"[ConfigTester] Error building the sweep job prompt: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def _sweep_control(request, action):
    try:
        data = await request.json()
        from . import workflow_sweep_runner as runner
        ok, state = getattr(runner, action)(data.get("sweep_id", ""))
        return web.json_response({"ok": ok, "state": state}, status=200 if ok else 409)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@server.PromptServer.instance.routes.post("/config_tester/sweep/pause")
async def sweep_pause(request):
    return await _sweep_control(request, "pause")

@server.PromptServer.instance.routes.post("/config_tester/sweep/resume")
async def sweep_resume(request):
    return await _sweep_control(request, "resume")

@server.PromptServer.instance.routes.post("/config_tester/sweep/cancel")
async def sweep_cancel(request):
    return await _sweep_control(request, "cancel")


def _create_image_links(source_dir, link_dir, items):
    """
    Create symlinks (or copies on Windows) from source images and videos into
    the output directory so ComfyUI's built-in /view endpoint can serve them.
    """
    # Use shared media extensions (images + videos) from the scanner module
    from .directory_scanner import MEDIA_EXTENSIONS

    # Collect all source files referenced by items
    filenames = set()
    for item in items:
        src = item.get("source_file")
        if src:
            filenames.add(src)

    # Scan source directory and subdirectories for matching files
    scan_dirs = [source_dir]
    images_subdir = os.path.join(source_dir, "images")
    if os.path.isdir(images_subdir):
        scan_dirs.append(images_subdir)

    for scan_dir in scan_dirs:
        try:
            for entry in os.scandir(scan_dir):
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in MEDIA_EXTENSIONS:
                    continue
                if entry.name not in filenames:
                    continue

                link_path = os.path.join(link_dir, entry.name)
                if os.path.exists(link_path):
                    continue

                try:
                    os.symlink(entry.path, link_path)
                except (OSError, NotImplementedError):
                    # Windows may need admin privileges for symlinks; fall back to copy
                    shutil.copy2(entry.path, link_path)
        except PermissionError:
            print(f"[DirScanner] Warning: Permission denied scanning {scan_dir}")


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "UltimateSamplerGrid": SamplerGridTester,
    "UltimateGridDashboard": SamplerConfigDashboardViewer,
    "UltimateConfigBuilder": UltimateConfigBuilder,
    "SmartJSONText": SmartJSONTextNode,
    "USCGSweepCollect": USCGSweepCollect,
    "USCGSweepCollectVideo": USCGSweepCollectVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSamplerGrid": "Ultimate Sampler Grid (Generator)",
    "UltimateGridDashboard": "Ultimate Grid Dashboard (Viewer)",
    "UltimateConfigBuilder": "Ultimate Config Builder",
    "SmartJSONText": "Smart JSON Text",
    "USCGSweepCollect": "USCG Sweep Collect",
    "USCGSweepCollectVideo": "USCG Sweep Collect Video",
}

WEB_DIRECTORY = "./web"