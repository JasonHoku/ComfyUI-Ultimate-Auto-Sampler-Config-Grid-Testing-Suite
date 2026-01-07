# ComfyUI Ultimate Auto Sampler Config Grid Testing Suite


<img width="1856" height="1030" alt="image" src="https://github.com/user-attachments/assets/e1d57553-80a8-4058-aea5-455e6bfbdf8a" />


**A professional-grade benchmarking and "IDE-like" testing suite for ComfyUI.**

Stop guessing which Sampler, Scheduler, or CFG value works best. This custom node suite allows you to generate massive Cartesian product grids, view them in an interactive infinite-canvas dashboard, and refine your settings with a "Revise & Generate" workflow without ever leaving the interface.

---

## 🌟 Key Features

### 🚀 Powerful Grid Generation
* **Cartesian Product Engine:** Automatically generates every permutation of your input settings. Test 2 Samplers × 3 Schedulers × 2 CFG scales in one go.
* **Smart Caching:** Intelligently skips model and LoRA reloading when consecutive runs share the same resources, making generation instant for parameter tweaks.
* **VAE Batching:** Includes a `vae_batch_size` input to batch decode images, significantly speeding up large grid runs.

### 🎨 Interactive Dashboard (The "IDE")
* **Infinite Canvas:** Pan and Zoom (Google Maps style) to inspect hundreds of images easily.
* **Smart Filtering:** Toggle visibility by Sampler, Scheduler, or LoRA type.
* **Sorting:** Instantly sort your grid by **Newest** or **Fastest** (generation time).
* **Session Management:** Save and Load previous testing sessions directly from the UI.

### ⚡ The "Revise & Generate" Workflow
* **One-Click Revision:** Click "REVISE" on any image to open a detail view.
* **Instant Tweak:** Adjust CFG, Steps, or Sampler for *just that specific image*.
* **Generate New:** A "GENERATE NEW" button queues the new variation immediately without needing to disconnect wires or change the main batch.
* **Similarity Reel:** The revision modal shows a side-scrolling reel of all other images that share the same seed, allowing for perfect A/B comparison.

### 🧹 Curation & JSON Export
* **Rejection System:** Click the red **"✕"** on bad generations to hide them.
* **Dual JSON Bars:**
    * **Green Bar:** Automatically groups all *accepted* configs into a clean, optimized JSON array ready for copy-pasting.
    * **Red Bar:** Collects all *rejected* configs so you know exactly what settings to avoid.

---

## 📦 Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```

2.  Clone this repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite.git](https://github.com/YOUR_USERNAME/ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite.git)
    ```

3.  Restart ComfyUI.

---

## 🛠️ Usage Guide

### 1. The Nodes
This suite consists of two main nodes found under the `sampling/testing` category:

1.  **Sampler Grid Tester (Generator):** The engine. It handles model loading, grid generation, and saving.
2.  **Grid Dashboard (Viewer):** The display. It renders the HTML output.

**Basic Setup:**
* Add the **Generator** node.
* Connect your Checkpoint, CLIP, and VAE (optional, see "Hybrid Inputs" below).
* Add the **Viewer** node.
* Connect the `dashboard_html` output from the Generator to the input of the Viewer.

### 2. The JSON Configuration
The `configs_json` widget determines your grid. It accepts an array of objects. All fields support single values or arrays.

**Example:**
```json
[
  {
    "sampler": ["euler", "dpmpp_2m"],
    "scheduler": ["normal", "karras"],
    "steps": [20, 30],
    "cfg": [7.0, 8.0],
    "lora": "None",
    "str_model": 1.0,
    "str_clip": 1.0
  }
]
```
*This example generates 8 images (2 samplers × 2 schedulers × 2 steps × 1 cfg).*


## Here are some combos you can try!

## 🏆 Group 1: The "Gold Standards" (Reliable Realism)

*Tests the 5 most reliable industry-standard combinations.* 5 samplers x 2 schedulers x 2 step settings x 2 cfgs = 40 images

    [
      {
        "sampler": ["dpmpp_2m", "dpmpp_2m_sde", "euler", "uni_pc", "heun"],
        "scheduler": ["karras", "normal"],
        "steps": [25, 30],
        "cfg": [6.0, 7.0],
        "lora": "None",
        "str_model": 1.0,
        "str_clip": 1.0
      }
    ]
    

## 🎨 Group 2: Artistic & Painterly

*Tests 5 creative/soft combinations best for illustration and anime.* 5 samplers x 2 schedulers x 3 step settings x 3 cfgs = 90 images

    [
      {
        "sampler": ["euler_ancestral", "dpmpp_sde", "dpmpp_2s_ancestral", "restart", "lms"],
        "scheduler": ["normal", "karras"],
        "steps": [20, 30, 40],
        "cfg": [5.0, 6.0, 7.0],
        "lora": "None",
        "str_model": 1.0,
        "str_clip": 1.0
      }
    ]
    

## ⚡ Group 3: Speed / Turbo / LCM

*Tests 4 ultra-fast configs. (Note: Ensure you are using a Turbo/LCM capable model or LoRA).* 4 samplers x 3 schedulers x 4 step settings x 2 cfgs = 96 images

    [
      {
        "sampler": ["lcm", "euler", "dpmpp_sde", "euler_ancestral"],
        "scheduler": ["simple", "sgm_uniform", "karras"],
        "steps": [4, 5, 6, 8],
        "cfg": [1.0, 1.5],
        "lora": "None",
        "str_model": 1.0,
        "str_clip": 1.0
      }
    ]
    

## 🦾 Group 4: Flux & SD3 Specials

*Tests 4 configs specifically tuned for newer Rectified Flow models like Flux and SD3.* 2 samplers x 3 schedulers x 3 step settings x 2 cfgs = 36 images

    [
      {
        "sampler": ["euler", "dpmpp_2m"],
        "scheduler": ["simple", "beta", "normal"],
        "steps": [20, 25, 30],
        "cfg": [1.0, 4.5],
        "lora": "None",
        "str_model": 1.0,
        "str_clip": 1.0
      }
    ]
    

## 🧪 Group 5: Experimental & Unique

*Tests 6 weird/niche combinations for discovering unique textures.* 6 samplers x 4 schedulers x 5 step settings x 4 cfgs = 480 images

    [
      {
        "sampler": ["dpmpp_3m_sde", "ddim", "ipndm", "heunpp2", "dpm_2_ancestral", "euler"],
        "scheduler": ["exponential", "normal", "karras", "beta"],
        "steps": [25, 30, 35, 40, 50],
        "cfg": [4.5, 6.0, 7.0, 8.0],
        "lora": "None",
        "str_model": 1.0,
        "str_clip": 1.0
      }
    ]

    

### 3. Hybrid Inputs (Optional)
The Generator node features built-in widgets for Model Selection and Prompts, but also has **Optional Inputs** for flexibility:
* **Standalone Mode:** Use the dropdown menu to select a checkpoint and type prompts into the text boxes.
* **Hybrid Mode:** Connect a `MODEL`, `CLIP`, `VAE`, or `CONDITIONING` wire. The node will automatically ignore the internal widget and use the connected input instead.

### 4. Performance Tuning (`vae_batch_size`)
* **Default (4):** Good for 8GB-12GB VRAM cards.
* **High (12-24):** For 24GB+ VRAM cards. Decodes many images at once for speed.
* **Unlimited (-1):** Decodes ALL images for a resolution at once. Fastest, but risks Out of Memory (OOM) errors on large batches.

---

## 🖥️ Dashboard Interface

### Toolbar
* **Session Input:** Type a session name (folder name) and click **LOAD** to view previous results without re-generating.
* **Sort:** Toggle between sorting by ID (Newest) or Duration (Fastest).
* **Filters:** Click the colored buttons (Green/Blue/Orange) to toggle visibility of specific parameters.

### JSON Bars (Bottom)
* **Green Bar (Accepted):** Contains a "Smart Grouped" JSON of all currently visible images. Click to select all, then copy-paste back into the `configs_json` widget to refine your batch.
* **Red Bar (Rejected):** Contains the configs of images you deleted with the **"✕"** button.

### Revision Modal
Clicking **REVISE** on a card opens the studio view:
1.  **Left:** Full-resolution preview.
2.  **Right:** Input fields to tweak settings for *this specific image*.
3.  **Bottom:** "Related Variants" reel showing other images with the same seed.
4.  **GENERATE NEW:** Queues the specific config you just edited.

---

## ⚠️ Troubleshooting

* **"Session not found":** Ensure the `session_name` matches a folder inside `ComfyUI/output/benchmarks/`.
* **Browser Zoom:** The dashboard uses an infinite canvas. If scrolling zooms the entire webpage instead of the canvas, ensure your mouse is hovering over the grid area.
* **OOM Errors:** If you crash during decoding, lower the `vae_batch_size` to 1 or 2.

---

## 📜 License

MIT License. Feel free to use, modify, and distribute.
