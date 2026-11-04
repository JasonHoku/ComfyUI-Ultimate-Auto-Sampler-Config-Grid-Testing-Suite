import re
import os
import json
import folder_paths
from .html_generator import get_html_template

# The template's <input id="session-input" class="session-name-input" type="text" value="__TITLE__">.
# (The old pattern expected class="session-input" right after the id and never matched, so
# update_session_name was always empty.)
SESSION_INPUT_RE = re.compile(r'<input[^>]*\bid="session-input"[^>]*?\bvalue="([^"]*)"')


class SamplerConfigDashboardViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "session_name": ("STRING", {"default": "my_session"}),
            },
            "optional": {
                "dashboard_html": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "sampling/testing"

    def view(self, session_name, unique_id, dashboard_html=None):
        # 1. If connected to Sampler (Fresh Generation)
        if dashboard_html:
            # --- MAGIC: EXTRACT SESSION NAME FROM HTML ---
            match = SESSION_INPUT_RE.search(dashboard_html)
            found_session = match.group(1) if match else None
            
            # --- CRITICAL FIX: INJECT CORRECT DASHBOARD NODE ID ---
            # The HTML coming from Sampler might have the Sampler's ID or a placeholder.
            # We must overwrite it with THIS dashboard node's ID so listeners work.
            # We look for the const TARGET_NODE_ID = "..." line in the JS.
            fixed_html = re.sub(
                r'const TARGET_NODE_ID = ".*?";', 
                f'const TARGET_NODE_ID = "{unique_id}";', 
                dashboard_html
            )
            
            return {
                "ui": {
                    "html": [fixed_html],
                    "update_session_name": [found_session] if found_session else []
                }
            }
        
        # 2. View Mode (Load from Disk)
        session_name = re.sub(r'[^\w\-]', '', session_name)
        if not session_name: session_name = "default_session"

        # The page is a shell: the dashboard fetches the manifest itself (/config_tester/manifest),
        # so nothing is read here. An unknown session just shows the landing page.
        empty_manifest = {"items": [], "meta": {"model": "", "positive": "", "negative": ""}, "session_name": session_name}
        html = get_html_template(session_name, empty_manifest, unique_id)
        return {"ui": {"html": [html]}}
