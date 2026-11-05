
### **File Structure Overview**

The refactoring splits the original monolithic `config_builder.js` into four modular files to separate concerns:

1. **`conf-builder-utilities.mjs`**: Contains core logic, state-independent helpers, data fetching, and caching mechanisms.
2. **`conf-builder-ui-components.mjs`**: Contains reusable UI component generators (sliders, dropdowns) and CSS styles.
3. **`conf-builder-config-management.mjs`**: Handles the specific business logic for rendering the Config Builder interface, including section rendering (Session, Config, Models, LoRAs) and complex interaction logic (Modals, State updates).
4. **`conf-builder-main.js`**: The entry point that registers the extension with ComfyUI, manages the node lifecycle, and ties the other modules together.

---

### **1. `conf-builder-utilities.mjs**`

**Role:** The backend logic and data layer. It handles communication with the ComfyUI server (`/object_info`, `/configbuilder/*`), manages data caching to prevent API spam, and performs pure data transformations (normalization, parsing).

#### **Major Functions & Sections:**

* **Cache Management**:
* `clearAllCaches()`: Resets all internal data stores (LoRAs, Models) to null.
* `refreshAllConfigBuilders()`: Iterates through all active nodes and triggers a re-render after clearing caches.


* **Path Normalization**:
* `normalizePath(path)`: Converts Windows backslashes to standard forward slashes for cross-platform consistency.
* `getShortName(path)`: Extracts a readable filename from a long path.


* **Data Fetching**:
* `getAvailableLoras()` / `getAvailableModels()`: Fetches the list of available files from ComfyUI's object info API and caches them.
* `getLoraFolders()` / `getModelFolders()`: Parses the flat file lists into a hierarchical folder structure.
* `getAvailableSessions()` / `getAvailableConfigs()`: Fetches saved session metadata and JSON config files from the backend.


* **Parsing Logic**:
* `parseLoraString(str)`: Converts a string like `"lora.safetensors:1.0:0.5"` into an object `{name, model_str, clip_str}`.
* `buildLoraString(...)`: The inverse of parsing; constructs the formatted string.


* **Iteration Calculation**:
* `getIterationCount(configArray)`: A critical logic function that calculates the total number of generation steps by multiplying the counts of models, LoRAs, samplers, schedulers, etc. It handles complex logic for folders and wildcards.
* `loraEntryCount(configArray, availableLoras)`: The LoRA factor of that product. It mirrors `process_lora_array` in `config_builder_node.py`: stacked LoRAs become one entry per combination of their strengths, folders add their file count times their strength variants.


* **LoRA Strength Rows** (the ＋ button):
* State key `config_arrays[i].lora_extra_strengths`: `{ "<lora name>": [ {"model": 0.6, "clip": 0.6}, ... ] }`. Only the EXTRA rows are stored; row 0 stays in `loras[j]` (`"name:m:c"`). Keyed by LoRA name like `lora_bypass_states` / `lora_strength_lock`. Values are rounded to 2 decimals, at most 7 extra rows (8 strengths in total). When `lora_strength_lock[name]` is not `false`, Python forces `clip = model`.
* Python emits one concrete string per row on the config's `lora` list (`"a:1.00:1.00"`, `"a:0.60:0.60"`, ...), and every combination for stacks. A non-empty entry replaces the old `lora_weight_arrays` entries of that name; states without the key behave exactly as before.
* `normalizeExtraStrengths(...)` is the JS twin of `_clean_extra_strengths` (validation, cap, duplicate flags). `legacyWeightArraysToExtras(...)` converts old Compare Strengths data (`lora_weight_arrays`) into rows on the first edit.


* **Config Conversion**:
* The state→configs_json transformation has moved to Python (`UltimateConfigBuilder.state_to_configs_json` in `config_builder_node.py`). The Builder UI's preview panel POSTs the state to `/configbuilder/preview` and renders the response. There is no JS-side equivalent of the old `convertStateToConfigs()` — see `docs/superpowers/specs/2026-04-28-builder-ui-sync-elimination-design.md`.
* `convertConfigsToConfigArrays(configs)`: Parses a loaded JSON file back into the internal state format for the UI (used by the "Load Session" path).



---

### **2. `conf-builder-ui-components.mjs**`

**Role:** The presentation layer library. It provides pure functions to create DOM elements. These functions are stateless and reusable.

#### **Major Functions & Sections:**

* **`createSearchableSelect(...)`**:
* Builds a custom dropdown with a text input for fuzzy searching through large lists (like 1000+ LoRAs).
* Handles keyboard navigation (Enter, Escape) and click-outside closing logic.


* **`createSlider(...)`**:
* Creates a dual-input control: a range slider and a numerical input box.
* Includes logic to sync both inputs bi-directionally (changing the slider updates the number box and vice versa).


* **`createInputGroup(...)`**:
* A layout helper that wraps an input element with a label and standard container styling.


* **`getStyles()`**:
* Returns a template string containing all the CSS for the extension. This includes the dark theme colors (`#1a1a1a`), flex grid layouts, and color-coded borders for different card types.



---

### **3. `conf-builder-config-management.mjs**`

**Role:** The UI controller. It orchestrates the rendering of the specific "Config Builder" interface. It imports helpers from *Utilities* and components from *UI Components* to build the actual application interface.

#### **Major Functions & Sections:**

* **Session & Config Rendering**:
* `renderSessionSection(...)`: Builds the top UI bar for naming the session and loading existing ComfyUI sessions.
* `renderConfigSection(...)`: Builds the controls for saving/loading JSON presets and toggling "Auto-Save".


* **Grid Rendering**:
* `createConfigArrayElement(...)`: Renders a single "Config Block" (a grouping of settings). It includes the main parameter inputs (Steps, CFG) and the container for Models and LoRAs.
* `renderModelsSection(...)`: Manages the list of models within a config block. It handles the collapsible header and the loop to render individual model cards.
* `renderLorasSection(...)`: Similar to models, but for LoRAs. It includes extra logic for handling the "Omit Triggers" section.


* **Item Cards**:
* `createModelElement(...)` / `createLoraElement(...)`: These generate the individual cards for a selected model or LoRA. They handle the specific logic for that item, such as "Type" selection (File vs. Folder) and the "Expand Folder" button logic.


* **Modal Logic**:
* `showTriggerLookupModal(...)`: Creates a popup overlay that fetches trigger words from a backend API (CivitAI) and allows users to select which words to ignore.


* **Main Render Loop**:
* `renderUI(...)`: The master function that clears the container and rebuilds the entire DOM tree based on the current state. It calls all the sub-renderers above.



---

### **4. `conf-builder-main.js**`

**Role:** The integration layer. It interfaces with the ComfyUI API (`app.registerExtension`) and manages the node's lifecycle.

#### **Major Functions & Sections:**

* **Node Registration (`app.registerExtension`)**:
* Defines `UltimateConfigBuilder.CompleteHTML`.
* Intercepts the creation of the `UltimateConfigBuilder` node.


* **`onNodeCreated`**:
* Hides the default text widget (`configWidget`).
* Initializes the `this.state` (the source of truth for the configuration) and `this.uiState` (for UI toggles like collapsed sections).
* Sets up data synchronization:
* `saveState`: Serializes `this.state` into the hidden widget so ComfyUI can save it in the workflow.
* `loadConfigFromBackend` / `saveConfigToBackend`: Handles AJAX requests to the server.


* Triggers the initial `renderUI()` call.


* **Global Refresh Hooks (`setup`)**:
* Monkey-patches `app.refreshComboInNodes` and `window.fetch`.
* This ensures that when a user adds a new model to ComfyUI and hits "Refresh", this custom node detects the event, clears its internal caches (via `utilities`), and re-renders to show the new files immediately.