import os
import json

def get_html_template(title, manifest_data, node_id, standalone=False):
    # 1. Normalize Data
    if isinstance(manifest_data, list):
        manifest_data = {"items": manifest_data, "meta": {"model": "", "positive": "", "negative": ""}}
    
    # Indent JSON for readability
    json_str = json.dumps(manifest_data, indent=2)
    # The manifest is inlined inside a <script> element: a note/prompt containing "</script>" would
    # end it early (page break / script injection). "<\/" is an equivalent JSON escape, so
    # JSON.parse (web/dashboard.js's incremental refresh) still reads the original text.
    json_str = json_str.replace("</", "<\\/")

    # 2. Resolve File Paths
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.join(current_dir, "resources")
    
    # Define JS Load Order (Dependencies first!)
    js_files = [
        "logic_state.js",    # Global variables
        "logic_utils.js",    # API & Helper functions
        "logic_ratings.js",  # Pairwise review: pure logic (normalisers, Bradley-Terry, pairing, scoreboard)
        "logic_ui.js",       # DOM Creation, Modals, JSON Bars
        "logic_upscale_modal.js",  # Upscale Modal (preset management, async upscale)
        "logic_virtual.js",  # The Virtual Scroll Engine & Viewport
        "logic_pipeline.js", # Sorting & Filtering Logic
        "logic_compare.js",  # Pairwise review: compare overlay, votes, scoreboard modal
        "logic_events.js",   # Event Listeners (Message, Keyboard)
        "logic_init.js",     # Initialization
        "logic_standalone.js"  # Standalone-page mode (own websocket, run API); must stay LAST
    ]

    try:
        # Load HTML
        with open(os.path.join(resource_dir, "template.html"), "r", encoding="utf-8") as f: 
            html_template = f.read()
        
        # Load CSS
        with open(os.path.join(resource_dir, "report.css"), "r", encoding="utf-8") as f: 
            css_content = f.read()
            
        # Load and Concatenate JS
        js_content = ""
        for js_file in js_files:
            file_path = os.path.join(resource_dir, js_file)
            with open(file_path, "r", encoding="utf-8") as f:
                js_content += f"\n/* --- {js_file} --- */\n"
                js_content += f.read() + "\n"

    except FileNotFoundError as e:
        return f"<h1>Error: Resource files not found.</h1><p>{e}</p>"

    # 3. Inject Data
    wrapped_json = f"/*__JSON_START__*/\nlet fullManifest = {json_str};\n/*__JSON_END__*/"

    # 4. Final Replacements
    final_html = html_template \
        .replace("__TITLE__", str(title)) \
        .replace("__NODE_ID__", str(node_id)) \
        .replace("__STANDALONE_FLAG__", "true" if standalone else "false") \
        .replace("let fullManifest = __JSON_DATA__;", wrapped_json) \
        .replace("__CSS_CONTENT__", css_content) \
        .replace("__JS_CONTENT__", js_content)

    return final_html