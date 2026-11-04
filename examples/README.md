# Example workflows

One ready-to-run workflow per model family. Each file is the same three-node graph (Config Builder -> Sampler Grid -> Grid Dashboard) with the Builder already filled in for that model: model, text encoder, CLIP type, VAE, sampler/scheduler/steps/cfg axes and a prompt.

## How to load one

1. Download the model files listed for the example and put each in the folder shown (under `ComfyUI/models/`). The official Comfy-Org workflow templates and Hugging Face repos for each family ship exactly these files.
2. Drag the `.json` onto the ComfyUI canvas.
3. Open the Config Builder. If your file names differ, pick yours in the model / text encoder / VAE dropdowns. A file you do not have is shown as missing rather than silently replaced.
4. Queue the prompt. Results appear in the Dashboard node.

Times are approximate on an RTX 5060 Ti 16 GB, not counting the first model load.

## SDXL / Illustrious (`uscg-sdxl-illustrious.json`)

| File | Folder |
|---|---|
| `sd_xl_base_1.0.safetensors` | `models/checkpoints/` |

* **What the grid tests:** 2 samplers (dpmpp_2m, euler) x 2 schedulers (karras, normal) x cfg 5 and 7 x 2 prompts at 25 steps, 1024x1024. **16 images**
* **Time:** TBD
* **Notes:** the checkpoint has the VAE and CLIP built in, so no extra files. Swap in any Illustrious/SDXL checkpoint.

## Anima (`uscg-anima.json`)

| File | Folder |
|---|---|
| `anima-base-v1.0.safetensors` | `models/diffusion_models/` |
| `qwen_3_06b_base.safetensors` | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |

* **What the grid tests:** 2 samplers (er_sde, euler) x simple scheduler x cfg 4 and 5 at 30 steps, 1024x1024. **4 images**
* **Time:** TBD
* **Notes:** the CLIP type stays `stable_diffusion`; there is no separate Anima option and ComfyUI picks the Anima text encoder from the Qwen3-0.6B file.

## Z-Image Turbo (`uscg-z-image-turbo.json`)

| File | Folder |
|---|---|
| `z_image_turbo_bf16.safetensors` | `models/diffusion_models/` |
| `qwen_3_4b.safetensors` | `models/text_encoders/` |
| `ae.safetensors` | `models/vae/` |

* **What the grid tests:** res_multistep / simple at 8 and 10 steps, cfg 1, 1024x1024. **2 images**
* **Time:** TBD
* **Notes:** CLIP type `lumina2`. Model Sampling is set to AuraFlow with shift 3, as in the official template.

## Krea 2 Turbo (`uscg-krea2-turbo.json`)

| File | Folder |
|---|---|
| `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models/` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |

* **What the grid tests:** euler / simple at 6 and 8 steps, cfg 1, 1024x1024. **2 images**
* **Time:** TBD
* **Notes:** CLIP type `krea2`.

## Flux.2 Klein 9B (`uscg-flux2-klein-9b.json`)

| File | Folder |
|---|---|
| `flux-2-klein-9b-fp8.safetensors` | `models/diffusion_models/` |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

* **What the grid tests:** euler at 4 and 6 steps, 1024x1024, each with the Advanced Sampling pipeline (CFG guider + Flux2 scheduler) and without it (plain KSampler, simple scheduler): the Builder always grids Advanced Sampling on vs off. **4 images**
* **Time:** TBD
* **Notes:** Klein is distilled, so cfg is 1; the official (distilled) template uses the CFG guider plus the Flux2 scheduler, which is the Advanced Sampling half of the grid. CLIP type `flux2`.
* **Base variant:** this example uses the distilled checkpoint. For the base model use `flux-2-klein-base-9b-fp8.safetensors` (same folder) with ~20 steps and cfg 5 (CFG guider + Flux2 scheduler), as in Comfy's `image_flux2_klein_image_edit_9b_base` template.

## MOSS-Image (`uscg-moss-image.json`)

| File | Folder |
|---|---|
| `moss_image_sft_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3.5_9b_w4a8_gptq.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

* **What the grid tests:** two config arrays with the same prompt and settings (er_sde / simple, 25 steps, cfg 2.5): one with Skip Layer Guidance on, one off. **2 images**
* **Time:** TBD
* **Notes:** SLG uses the official recipe, single block 22 for the first 0-50 % of the steps. Compare the two images to see what it does. CLIP type `moss_image`. A smaller `flux2-vae_small.safetensors` VAE also exists and can be picked in the Builder instead. The 9B text encoder is large; on a 16 GB card expect model swapping.

## Make your own

* Edit the Builder: add LoRAs or a whole LoRA folder, more samplers, schedulers, step counts, cfg values, resolutions or prompts. Every list is multiplied together.
* Raise **Additional Random Seeds** in Run Settings to render extra seeds per config, handy for telling what the model does from what a LoRA does.
* Add another config array to compare models or settings side by side (the MOSS example does this for SLG).
