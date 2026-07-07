"""
Image Generation and Sampling Module
Handles the core image generation, decoding, and batch management.

============================================================================
🚨 CRITICAL: torch.inference_mode() WRAPPING — DO NOT REMOVE 🚨
============================================================================
Every top-level GPU-inference entry point in this module is decorated with
`@torch.inference_mode()`. This mirrors ComfyUI's PromptExecutor which wraps
every node execution in inference_mode (ComfyUI/execution.py:732).

WHY THIS MATTERS:
  We are an out-of-graph caller — we invoke ComfyUI's sampling / VAE / upscale
  models directly from a background thread (the dashboard upscale runner) and
  from our orchestrator, bypassing the normal prompt executor. Without
  inference_mode wrapping, autograd's version-counter and reference-keeping
  machinery stays active. For diffusion sampling, beam-search generation,
  VAE encode/decode, and upscale-model inference, that means intermediate
  activations and past_key_values can't be released — measured as a 6× VRAM
  blowup on Florence2 (29GB on a 16GB card vs <5GB in the standalone workflow
  on the same image, May 2026). Same pattern applies to SDXL sampling,
  SeedVR2 diffusion, and ESRGAN-style upscalers, just at a smaller magnitude.

  torch.inference_mode() is STRICTER than torch.no_grad():
    - no_grad: disables gradient tracking, but version counters and autograd
      machinery stay partly active
    - inference_mode: tensors are entirely outside autograd, true read-only —
      this is what lets HF generate() actually release KV cache between beam
      steps and what lets ComfyUI's sample loop reclaim activations between
      diffusion steps

WHERE IT'S APPLIED:
  - @torch.inference_mode() on generate_image, upscale_image,
    decode_latent_with_vae, seedvr2_upscale (this file)
  - Inside _call_node() in ltx_video_generation.py (covers all 25 ComfyUI
    node invocations across florence2_hires.py + ltx_video_generation.py)

IF YOU ADD A NEW INFERENCE ENTRY POINT:
  Wrap it in @torch.inference_mode() (decorator) or
  `with torch.inference_mode():` (context). Symptom of missing it: GPU OOM
  on workloads that work fine in a standalone ComfyUI workflow, or sudden
  VRAM growth that doesn't match the model size. See
  ComfyUI/execution.py:732 for the canonical pattern.
============================================================================
"""

import time
import os
import re
import json
import torch
import nodes
import numpy as np
from PIL import Image


# NOTE: torch.inference_mode() decorator — see module docstring above for why
# this is REQUIRED on all out-of-graph inference entry points. Removing it
# will reintroduce the same 6× VRAM blowup we fixed in May 2026.
@torch.inference_mode()
def generate_image(
    patched_model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive_conditioning,
    negative_conditioning,
    latent_input,
    denoise,
    attention_mode="default",
    model_sampling_override="none",
    model_sampling_shift=1.73,
    model_sampling_flux_max_shift=1.15,
    model_sampling_flux_base_shift=0.5,
    use_advanced_sampling=False,
    advanced_guider="cfg_guider",
    advanced_scheduler="basic",
    flux_guidance_value=0.0,
    use_deep_shrink=False,
    deep_shrink_block_number=3,
    deep_shrink_downscale_factor=2.0,
    deep_shrink_start_percent=0.0,
    deep_shrink_end_percent=0.35,
    deep_shrink_downscale_after_skip=True,
    deep_shrink_downscale_method="bicubic",
    deep_shrink_upscale_method="bicubic",
    width=1024,
    height=1024
):
    """
    Generate a single image using ComfyUI's KSampler or SamplerCustomAdvanced pipeline.

    Args:
        patched_model: Model (potentially with LoRAs applied)
        seed: Random seed
        steps: Number of sampling steps
        cfg: CFG scale
        sampler_name: Sampler name (e.g., "euler", "dpmpp_2m")
        scheduler: Scheduler name (e.g., "normal", "karras")
        positive_conditioning: Positive conditioning tensor
        negative_conditioning: Negative conditioning tensor
        latent_input: Input latent tensor dict
        denoise: Denoise strength (0.0-1.0)
        attention_mode: Attention implementation to use ("default", "xformers", "pytorch",
                       "flash", "sage", "sage3", "sub_quad", "split")
        model_sampling_override: Model sampling patch type ("none", "aura_flow", "flux", "sd3")
        model_sampling_shift: Shift value for AuraFlow/SD3 model sampling
        model_sampling_flux_max_shift: Max shift for Flux model sampling
        model_sampling_flux_base_shift: Base shift for Flux model sampling
        use_advanced_sampling: Whether to use SamplerCustomAdvanced pipeline
        advanced_guider: Guider type ("cfg_guider", "basic_guider")
        advanced_scheduler: Scheduler type for advanced sampling ("basic", "flux2")
        flux_guidance_value: Flux guidance value (0.0 = disabled)
        use_deep_shrink: Whether to apply Kohya Deep Shrink (PatchModelAddDownscale)
        deep_shrink_block_number: UNet block to downscale at (1-32, default 3)
        deep_shrink_downscale_factor: Downscale factor (0.1-9.0, default 2.0)
        deep_shrink_start_percent: Diffusion start % for the patch (default 0.0)
        deep_shrink_end_percent: Diffusion end % for the patch (default 0.35)
        deep_shrink_downscale_after_skip: Whether to downscale after skip (default True)
        deep_shrink_downscale_method: Downscale interpolation method (default "bicubic")
        deep_shrink_upscale_method: Upscale interpolation method (default "bicubic")
        width: Image width (used by Flux model sampling and Flux2Scheduler)
        height: Image height (used by Flux model sampling and Flux2Scheduler)

    Returns:
        tuple: (result_latent_dict, generation_duration_seconds)
    """
    # Apply attention mode override if not "default"
    original_attn_override = None
    if attention_mode and attention_mode != "default":
        try:
            from comfy.ldm.modules.attention import get_attention_function
            attn_func = get_attention_function(attention_mode, default=None)
            if attn_func is not None:
                # Save original override (if any) and set the new one
                original_attn_override = patched_model.model_options.get("transformer_options", {}).get("optimized_attention_override")
                if "transformer_options" not in patched_model.model_options:
                    patched_model.model_options["transformer_options"] = {}
                patched_model.model_options["transformer_options"]["optimized_attention_override"] = attn_func
            else:
                print(f"[GridTester] ⚠️ Attention mode '{attention_mode}' not available, using default")
        except Exception as e:
            print(f"[GridTester] ⚠️ Could not set attention mode '{attention_mode}': {e}")

    # === Model Sampling Override ===
    # Clone model and patch model_sampling if an override is requested.
    # This modifies the model's internal noise schedule for specific model families.
    if model_sampling_override and model_sampling_override != "none":
        import comfy.model_sampling
        patched_model = patched_model.clone()

        if model_sampling_override == "aura_flow":
            # AuraFlow/Qwen Image: discrete flow with shift, multiplier=1.0
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced(patched_model.model.model_config)
            model_sampling.set_parameters(shift=float(model_sampling_shift), multiplier=1.0)
            patched_model.add_object_patch("model_sampling", model_sampling)
            print(f"[GridTester] 🔧 Applied ModelSamplingAuraFlow (shift={model_sampling_shift})")

        elif model_sampling_override == "sd3":
            # SD3: discrete flow with shift, multiplier=1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced(patched_model.model.model_config)
            model_sampling.set_parameters(shift=float(model_sampling_shift), multiplier=1000)
            patched_model.add_object_patch("model_sampling", model_sampling)
            print(f"[GridTester] 🔧 Applied ModelSamplingSD3 (shift={model_sampling_shift})")

        elif model_sampling_override == "flux":
            # Flux: dynamic shift computed from image dimensions
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            max_s = float(model_sampling_flux_max_shift)
            base_s = float(model_sampling_flux_base_shift)
            x1, x2 = 256, 4096
            mm = (max_s - base_s) / (x2 - x1)
            b = base_s - mm * x1
            shift = (width * height / (8 * 8 * 2 * 2)) * mm + b
            model_sampling = ModelSamplingAdvanced(patched_model.model.model_config)
            model_sampling.set_parameters(shift=shift)
            patched_model.add_object_patch("model_sampling", model_sampling)
            print(f"[GridTester] 🔧 Applied ModelSamplingFlux (max_shift={max_s}, base_shift={base_s}, computed_shift={shift:.4f})")

        elif model_sampling_override == "flux2":
            # Flux2: uses ModelSamplingFlux with fixed shift (default 2.02, matching ComfyUI's Flux2 supported_models)
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            shift = float(model_sampling_shift) if model_sampling_shift else 2.02
            model_sampling = ModelSamplingAdvanced(patched_model.model.model_config)
            model_sampling.set_parameters(shift=shift)
            patched_model.add_object_patch("model_sampling", model_sampling)
            print(f"[GridTester] 🔧 Applied ModelSamplingFlux2 (shift={shift})")

    # === Kohya Deep Shrink (PatchModelAddDownscale) ===
    # Patches the UNet to downscale features at a specific block during the
    # early portion of diffusion. Wraps ComfyUI's built-in
    # PatchModelAddDownscale (comfy_extras/nodes_model_downscale.py).
    # Looked up via NODE_CLASS_MAPPINGS per project convention.
    #
    # Current ComfyUI ships this as a V3 io.ComfyNode subclass with a
    # classmethod execute() returning io.NodeOutput (whose positional outputs
    # live in .args). Older builds had a V1 instance method `.patch()`. We
    # dispatch to either depending on what we find — same pattern as
    # ltx_video_generation._call_node / _unwrap.
    if use_deep_shrink:
        try:
            deep_shrink_cls = nodes.NODE_CLASS_MAPPINGS.get("PatchModelAddDownscale")
            if deep_shrink_cls is not None:
                ds_kwargs = dict(
                    model=patched_model,
                    block_number=int(deep_shrink_block_number),
                    downscale_factor=float(deep_shrink_downscale_factor),
                    start_percent=float(deep_shrink_start_percent),
                    end_percent=float(deep_shrink_end_percent),
                    downscale_after_skip=bool(deep_shrink_downscale_after_skip),
                    downscale_method=str(deep_shrink_downscale_method),
                    upscale_method=str(deep_shrink_upscale_method),
                )
                # Try V3 first (classmethod execute, NodeOutput return).
                ds_result = None
                try:
                    from comfy_api.latest import io as _ds_io
                    if isinstance(deep_shrink_cls, type) and issubclass(deep_shrink_cls, _ds_io.ComfyNode):
                        ds_result = deep_shrink_cls.execute(**ds_kwargs)
                except (ImportError, TypeError):
                    pass
                if ds_result is None:
                    # V1 fallback (instance method named by FUNCTION, default "patch").
                    fn_name = getattr(deep_shrink_cls, "FUNCTION", "patch")
                    ds_result = getattr(deep_shrink_cls(), fn_name)(**ds_kwargs)
                # Unwrap to a single MODEL value.
                if hasattr(ds_result, "args") and isinstance(getattr(ds_result, "args"), tuple):
                    patched_model = ds_result.args[0]  # V3 NodeOutput
                elif isinstance(ds_result, (tuple, list)):
                    patched_model = ds_result[0]      # V1 tuple
                else:
                    patched_model = ds_result          # bare model
                print(f"[GridTester] 🔧 Applied PatchModelAddDownscale (Kohya Deep Shrink): block={deep_shrink_block_number}, factor={deep_shrink_downscale_factor}, range={deep_shrink_start_percent}-{deep_shrink_end_percent}, after_skip={deep_shrink_downscale_after_skip}, down={deep_shrink_downscale_method}, up={deep_shrink_upscale_method}")
            else:
                print("[GridTester] ⚠️ PatchModelAddDownscale node not available in this ComfyUI build — Deep Shrink skipped")
        except Exception as e:
            print(f"[GridTester] ⚠️ Failed to apply Deep Shrink: {e}")

    # === Flux Guidance ===
    # Modify positive conditioning with guidance value (used by Flux 1 models)
    if flux_guidance_value and float(flux_guidance_value) > 0:
        import node_helpers
        positive_conditioning = node_helpers.conditioning_set_values(
            positive_conditioning, {"guidance": float(flux_guidance_value)}
        )
        print(f"[GridTester] 🔧 Applied FluxGuidance (guidance={flux_guidance_value})")

    t0 = time.time()

    try:
        if use_advanced_sampling:
            # === Advanced Sampling Pipeline (SamplerCustomAdvanced) ===
            import comfy.samplers
            import comfy.sample
            import comfy.sampler_helpers
            import comfy.model_management
            import comfy.utils
            import latent_preview

            # Noise
            class _Noise_RandomNoise:
                def __init__(self, seed):
                    self.seed = seed
                def generate_noise(self, input_latent):
                    latent_image = input_latent["samples"]
                    batch_inds = input_latent.get("batch_index")
                    return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

            noise = _Noise_RandomNoise(seed)

            # Guider
            if advanced_guider == "basic_guider":
                guider = comfy.samplers.CFGGuider(patched_model)
                guider.inner_set_conds({"positive": comfy.sampler_helpers.convert_cond(positive_conditioning)})
                guider.set_cfg(1.0)
                print(f"[GridTester] 🔧 Advanced sampling: BasicGuider (no CFG)")
            else:
                guider = comfy.samplers.CFGGuider(patched_model)
                guider.set_conds(positive_conditioning, negative_conditioning)
                guider.set_cfg(cfg)
                print(f"[GridTester] 🔧 Advanced sampling: CFGGuider (cfg={cfg})")

            # Sampler object
            sampler_obj = comfy.samplers.sampler_object(sampler_name)

            # Sigmas
            if advanced_scheduler == "flux2":
                import math

                def _flux2_generalized_time_snr_shift(t, mu, sigma):
                    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

                def _flux2_compute_empirical_mu(image_seq_len, num_steps):
                    a1, b1 = 8.73809524e-05, 1.89833333
                    a2, b2 = 0.00016927, 0.45666666
                    if image_seq_len > 4300:
                        return float(a2 * image_seq_len + b2)
                    m_200 = a2 * image_seq_len + b2
                    m_10 = a1 * image_seq_len + b1
                    a = (m_200 - m_10) / 190.0
                    b_val = m_200 - 200.0 * a
                    return float(a * num_steps + b_val)

                seq_len = round(width * height / (16 * 16))
                mu = _flux2_compute_empirical_mu(seq_len, steps)
                timesteps = torch.linspace(1, 0, steps + 1)
                sigmas = torch.FloatTensor([_flux2_generalized_time_snr_shift(t.item(), mu, 1.0) for t in timesteps])
                print(f"[GridTester] 🔧 Advanced sampling: Flux2Scheduler (steps={steps}, {width}x{height}, mu={mu:.4f})")
            else:
                # BasicScheduler
                total_steps = steps
                if denoise < 1.0:
                    if denoise <= 0.0:
                        sigmas = torch.FloatTensor([])
                    else:
                        total_steps = int(steps / denoise)
                model_sampling_obj = patched_model.get_model_object("model_sampling")
                sigmas = comfy.samplers.calculate_sigmas(model_sampling_obj, scheduler, total_steps).cpu()
                sigmas = sigmas[-(steps + 1):]
                print(f"[GridTester] 🔧 Advanced sampling: BasicScheduler ({scheduler}, {steps} steps)")

            # Execute SamplerCustomAdvanced
            latent = latent_input.copy()
            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(
                guider.model_patcher,
                latent_image,
                latent.get("downscale_ratio_spacial", None),
                latent.get("downscale_ratio_temporal", None),
            )
            latent["samples"] = latent_image

            noise_mask = latent.get("noise_mask")

            x0_output = {}
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(
                noise.generate_noise(latent), latent_image, sampler_obj, sigmas,
                denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
                seed=noise.seed
            )
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out["samples"] = samples
            result = ({"samples": out["samples"]},)

        else:
            # === Standard KSampler path ===
            result = nodes.common_ksampler(
                model=patched_model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_conditioning,
                negative=negative_conditioning,
                latent=latent_input,
                denoise=denoise
            )
    finally:
        # Restore original attention override
        if attention_mode and attention_mode != "default":
            if original_attn_override is not None:
                patched_model.model_options["transformer_options"]["optimized_attention_override"] = original_attn_override
            elif "transformer_options" in patched_model.model_options and "optimized_attention_override" in patched_model.model_options["transformer_options"]:
                del patched_model.model_options["transformer_options"]["optimized_attention_override"]

    duration = round(time.time() - t0, 3)

    return result[0], duration


def tiled_hires_sample(latent_input, patched_model, config, positive_conditioning, negative_conditioning,
                       hires_steps, hires_denoise, tile_width, tile_height, mask_blur, tile_padding,
                       force_uniform, pixel_width, pixel_height):
    """
    Run HiRes fix sampling in tiles to prevent OOM on large images.
    Splits the latent into overlapping tiles, samples each, then blends them back together.

    Args:
        latent_input: Dict with "samples" key — the upscaled latent to denoise
        patched_model: Model for sampling
        config: Generation config (seed, cfg, sampler, scheduler)
        positive_conditioning, negative_conditioning: Conditioning tensors
        hires_steps: Number of sampling steps
        hires_denoise: Denoise strength
        tile_width, tile_height: Tile size in pixels (will be converted to latent space /8)
        mask_blur: Gaussian blur radius for tile seam blending (pixels)
        tile_padding: Extra context padding around each tile (pixels)
        force_uniform: If True, force all tiles to be the same size (may crop edges)
        pixel_width, pixel_height: Full image pixel dimensions (for logging)

    Returns:
        dict: Result latent dict with "samples" key
    """
    import torch

    samples = latent_input["samples"]
    # Convert pixel dimensions to latent space (8x smaller)
    # Handle both 4D [B, C, H, W] and 5D [B, C, T, H, W] latent formats (video VAEs add temporal dim)
    if samples.ndim == 5:
        lat_h, lat_w = samples.shape[3], samples.shape[4]
    else:
        lat_h, lat_w = samples.shape[2], samples.shape[3]
    tw = tile_width // 8
    th = tile_height // 8
    pad = tile_padding // 8
    blur = max(1, mask_blur // 8)  # Blur in latent space

    # Calculate tile grid
    def calc_tiles(total, tile_size, padding, uniform):
        """Calculate tile start positions with overlap = 2 * padding."""
        if total <= tile_size:
            return [(0, total)]
        stride = tile_size - 2 * padding
        if stride <= 0:
            stride = tile_size // 2
        tiles = []
        pos = 0
        while pos < total:
            end = min(pos + tile_size, total)
            if uniform and end == total and end - pos < tile_size:
                # Shift last tile back to maintain uniform size
                pos = max(0, total - tile_size)
                end = total
            tiles.append((pos, end))
            if end == total:
                break
            pos += stride
        return tiles

    x_tiles = calc_tiles(lat_w, tw, pad, force_uniform)
    y_tiles = calc_tiles(lat_h, th, pad, force_uniform)

    total_tiles = len(x_tiles) * len(y_tiles)
    print(f"[GridTester] 🔍 Tiled HiRes sampling: {len(x_tiles)}x{len(y_tiles)} = {total_tiles} tiles "
          f"(tile={tile_width}x{tile_height}px, padding={tile_padding}px, blur={mask_blur}px)")

    # Output accumulator with weighted blending
    result_samples = torch.zeros_like(samples)
    is_5d = samples.ndim == 5

    # Weight map shape matches spatial dims only
    if is_5d:
        weight_map = torch.zeros(1, 1, 1, lat_h, lat_w, device=samples.device)
    else:
        weight_map = torch.zeros(1, 1, lat_h, lat_w, device=samples.device)

    tile_idx = 0
    for yi, (y_start, y_end) in enumerate(y_tiles):
        for xi, (x_start, x_end) in enumerate(x_tiles):
            tile_idx += 1
            # Extract tile from latent (handle both 4D and 5D)
            if is_5d:
                tile_latent = samples[:, :, :, y_start:y_end, x_start:x_end].clone()
            else:
                tile_latent = samples[:, :, y_start:y_end, x_start:x_end].clone()

            print(f"[GridTester] 🔍   Tile {tile_idx}/{total_tiles}: latent region [{y_start}:{y_end}, {x_start}:{x_end}]")

            # Run KSampler on this tile
            tile_result, _ = generate_image(
                patched_model, config.get("seed", 0), hires_steps, config.get("cfg", 7),
                config.get("sampler", "euler"), config.get("scheduler", "normal"),
                positive_conditioning, negative_conditioning,
                {"samples": tile_latent}, hires_denoise,
                width=(x_end - x_start) * 8, height=(y_end - y_start) * 8
            )

            tile_out = tile_result["samples"]
            tile_h = y_end - y_start
            tile_w = x_end - x_start

            # Create feathered weight mask for this tile (higher weight in center, fading at edges)
            mask = torch.ones(tile_h, tile_w, device=samples.device)
            if blur > 0:
                # Feather edges: linear ramp over blur pixels
                for b in range(blur):
                    factor = (b + 1) / (blur + 1)
                    # Top edge
                    if y_start > 0 and b < tile_h:
                        mask[b, :] *= factor
                    # Bottom edge
                    if y_end < lat_h and b < tile_h:
                        mask[tile_h - 1 - b, :] *= factor
                    # Left edge
                    if x_start > 0 and b < tile_w:
                        mask[:, b] *= factor
                    # Right edge
                    if x_end < lat_w and b < tile_w:
                        mask[:, tile_w - 1 - b] *= factor

            # Accumulate weighted results (broadcast mask to match tensor dims)
            if is_5d:
                mask_shaped = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
                result_samples[:, :, :, y_start:y_end, x_start:x_end] += tile_out * mask_shaped
                weight_map[:, :, :, y_start:y_end, x_start:x_end] += mask_shaped
            else:
                mask_shaped = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                result_samples[:, :, y_start:y_end, x_start:x_end] += tile_out * mask_shaped
                weight_map[:, :, y_start:y_end, x_start:x_end] += mask_shaped

    # Normalize by weights to blend overlapping regions
    weight_map = torch.clamp(weight_map, min=1e-6)
    result_samples = result_samples / weight_map

    print(f"[GridTester] 🔍 Tiled HiRes sampling complete ({total_tiles} tiles)")
    return {"samples": result_samples}


# @torch.inference_mode() — see module docstring. Covers KSampler hires fix,
# ImageUpscaleWithModel (ESRGAN-style), and VAE encode/decode in this function.
@torch.inference_mode()
def upscale_image(result_latent, vae, patched_model, upscaling_config, config, positive_conditioning, negative_conditioning, width, height):
    """
    Apply upscaling to a generated latent based on upscaling settings.

    Args:
        result_latent: Generated latent dict with "samples" key
        vae: VAE model for encode/decode
        patched_model: Patched model for re-sampling (HiRes fix)
        upscaling_config: Dict with mode, upscale_ratio, hires_denoise, etc.
        config: Current generation config (steps, sampler, scheduler, etc.)
        positive_conditioning: Positive conditioning
        negative_conditioning: Negative conditioning
        width: Original image width
        height: Original image height

    Returns:
        tuple: (result, duration) where result is either a latent dict or PIL Image
    """
    import torch
    import time

    mode = upscaling_config.get("mode", "hires_only")
    upscale_ratio = float(upscaling_config.get("upscale_ratio", 1.5))
    hires_denoise = float(upscaling_config.get("hires_denoise", 0.5))
    hires_steps = int(upscaling_config.get("hires_steps", 0)) or config.get("steps", 20)
    tiled_vae = upscaling_config.get("tiled_vae", False)
    tile_size = int(upscaling_config.get("tile_size", 512))
    tile_overlap = int(upscaling_config.get("tile_overlap", 64))
    temporal_size = int(upscaling_config.get("temporal_size", 512))
    temporal_overlap = int(upscaling_config.get("temporal_overlap", 64))
    upscale_model_name = upscaling_config.get("upscale_model", "")
    upscale_size = float(upscaling_config.get("upscale_size", 2.0))
    resize_method = upscaling_config.get("resize_method", "bilinear")
    hires_tiled_sampling = upscaling_config.get("hires_tiled_sampling", False)
    hires_tile_width = int(upscaling_config.get("hires_tile_width", 512))
    hires_tile_height = int(upscaling_config.get("hires_tile_height", 512))
    hires_mask_blur = int(upscaling_config.get("hires_mask_blur", 8))
    hires_tile_padding = int(upscaling_config.get("hires_tile_padding", 32))
    hires_force_uniform_tiles = upscaling_config.get("hires_force_uniform_tiles", False)

    new_w = int(width * upscale_ratio)
    new_h = int(height * upscale_ratio)

    t0 = time.time()
    print(f"[GridTester] 🔍 Upscaling: mode={mode}, ratio={upscale_ratio}, target={new_w}x{new_h}")

    if mode == "hires_only":
        # Upscale latent in latent space → re-sample with denoise
        import comfy.utils
        latent_samples = result_latent["samples"]
        # Latent space is 8x smaller than pixel space
        upscaled_latent = comfy.utils.common_upscale(
            latent_samples, new_w // 8, new_h // 8, resize_method, "disabled"
        )

        if hires_tiled_sampling:
            hires_latent = tiled_hires_sample(
                {"samples": upscaled_latent}, patched_model, config,
                positive_conditioning, negative_conditioning,
                hires_steps, hires_denoise,
                hires_tile_width, hires_tile_height, hires_mask_blur, hires_tile_padding,
                hires_force_uniform_tiles, new_w, new_h
            )
        else:
            hires_latent, hires_duration = generate_image(
                patched_model, config.get("seed", 0), hires_steps, config.get("cfg", 7),
                config.get("sampler", "euler"), config.get("scheduler", "normal"),
                positive_conditioning, negative_conditioning,
                {"samples": upscaled_latent}, hires_denoise,
                width=new_w, height=new_h
            )

        duration = round(time.time() - t0, 3)
        print(f"[GridTester] 🔍 HiRes fix complete in {duration}s → {new_w}x{new_h}")
        return hires_latent, duration

    elif mode == "model_only":
        # Decode → model upscale → optional resize to target → return as PIL image
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
        import numpy as np

        # Use tiled VAE decode if enabled (prevents OOM on large images)
        if tiled_vae:
            from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
            vae.first_stage_model.tile_sample_min_size = tile_size
            vae.first_stage_model.tile_latent_min_size = tile_size // 8
            vae.first_stage_model.tile_overlap_factor = tile_overlap / tile_size if tile_size > 0 else 0.125
            if hasattr(vae.first_stage_model, 'tile_sample_min_size_temporal'):
                vae.first_stage_model.tile_sample_min_size_temporal = temporal_size
            if hasattr(vae.first_stage_model, 'tile_latent_min_size_temporal'):
                vae.first_stage_model.tile_latent_min_size_temporal = temporal_size // 8
            if hasattr(vae.first_stage_model, 'tile_overlap_factor_temporal'):
                vae.first_stage_model.tile_overlap_factor_temporal = temporal_overlap / temporal_size if temporal_size > 0 else 0.125
        pil_image = decode_latent_with_vae(vae, result_latent["samples"])

        img_np = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, C)

        loader = UpscaleModelLoader()
        (up_model,) = loader.load_model(upscale_model_name)
        upscaler = ImageUpscaleWithModel()
        (upscaled_tensor,) = upscaler.upscale(up_model, img_tensor)

        # If upscale_size differs from model's native scale, resize the output
        target_w = int(width * upscale_size)
        target_h = int(height * upscale_size)
        actual_h, actual_w = upscaled_tensor.shape[1], upscaled_tensor.shape[2]
        if abs(actual_w - target_w) > 4 or abs(actual_h - target_h) > 4:
            import comfy.utils
            # Resize from model's native output to user-specified upscale_size
            upscaled_tensor = upscaled_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW
            upscaled_tensor = comfy.utils.common_upscale(upscaled_tensor, target_w, target_h, resize_method, "disabled")
            upscaled_tensor = upscaled_tensor.permute(0, 2, 3, 1)  # NCHW → NHWC

        up_np = upscaled_tensor[0].cpu().float().numpy()
        up_np = np.clip(up_np * 255, 0, 255).astype(np.uint8)
        from PIL import Image as PILImage
        upscaled_image = PILImage.fromarray(up_np)

        duration = round(time.time() - t0, 3)
        print(f"[GridTester] 🔍 Model upscale complete in {duration}s → {upscaled_image.size[0]}x{upscaled_image.size[1]}")
        return upscaled_image, duration

    elif mode == "model_then_hires":
        # Model upscale first → optional resize → encode to latent → HiRes fix
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
        import numpy as np

        # Use tiled VAE decode if enabled
        if tiled_vae:
            vae.first_stage_model.tile_sample_min_size = tile_size
            vae.first_stage_model.tile_latent_min_size = tile_size // 8
            vae.first_stage_model.tile_overlap_factor = tile_overlap / tile_size if tile_size > 0 else 0.125
            if hasattr(vae.first_stage_model, 'tile_sample_min_size_temporal'):
                vae.first_stage_model.tile_sample_min_size_temporal = temporal_size
            if hasattr(vae.first_stage_model, 'tile_latent_min_size_temporal'):
                vae.first_stage_model.tile_latent_min_size_temporal = temporal_size // 8
            if hasattr(vae.first_stage_model, 'tile_overlap_factor_temporal'):
                vae.first_stage_model.tile_overlap_factor_temporal = temporal_overlap / temporal_size if temporal_size > 0 else 0.125
        pil_image = decode_latent_with_vae(vae, result_latent["samples"])

        img_np = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        loader = UpscaleModelLoader()
        (up_model,) = loader.load_model(upscale_model_name)
        upscaler = ImageUpscaleWithModel()
        (upscaled_tensor,) = upscaler.upscale(up_model, img_tensor)

        # If upscale_size differs from model's native scale, resize before HiRes fix
        target_w = int(width * upscale_size)
        target_h = int(height * upscale_size)
        actual_h, actual_w = upscaled_tensor.shape[1], upscaled_tensor.shape[2]
        if abs(actual_w - target_w) > 4 or abs(actual_h - target_h) > 4:
            import comfy.utils
            upscaled_tensor = upscaled_tensor.permute(0, 3, 1, 2)  # NHWC → NCHW
            upscaled_tensor = comfy.utils.common_upscale(upscaled_tensor, target_w, target_h, resize_method, "disabled")
            upscaled_tensor = upscaled_tensor.permute(0, 2, 3, 1)  # NCHW → NHWC

        up_h, up_w = upscaled_tensor.shape[1], upscaled_tensor.shape[2]

        # Encode back to latent space for HiRes fix
        encoded_latent = vae.encode(upscaled_tensor[:, :, :, :3])

        if hires_tiled_sampling:
            hires_latent = tiled_hires_sample(
                {"samples": encoded_latent}, patched_model, config,
                positive_conditioning, negative_conditioning,
                hires_steps, hires_denoise,
                hires_tile_width, hires_tile_height, hires_mask_blur, hires_tile_padding,
                hires_force_uniform_tiles, up_w, up_h
            )
        else:
            hires_latent, hires_duration = generate_image(
                patched_model, config.get("seed", 0), hires_steps, config.get("cfg", 7),
                config.get("sampler", "euler"), config.get("scheduler", "normal"),
                positive_conditioning, negative_conditioning,
                {"samples": encoded_latent}, hires_denoise,
                width=up_w, height=up_h
            )

        duration = round(time.time() - t0, 3)
        print(f"[GridTester] 🔍 Model+HiRes upscale complete in {duration}s → {up_w}x{up_h}")
        return hires_latent, duration

    else:
        print(f"[GridTester] ⚠️ Unknown upscale mode: {mode}")
        return result_latent, 0


# @torch.inference_mode() — see module docstring. VAE decode is a model forward
# pass and needs inference_mode to avoid pinning intermediate activations.
@torch.inference_mode()
def decode_latent_with_vae(vae, latent_samples):
    """
    Decode latent samples to pixel space using VAE.

    Args:
        vae: VAE model
        latent_samples: Latent tensor to decode

    Returns:
        PIL.Image: Decoded image
    """
    import torch

    # Check for NaN in input latent
    if torch.isnan(latent_samples).any():
        print(f"[GridTester] ⚠️ VAE decode: input latent contains NaN! shape={latent_samples.shape} dtype={latent_samples.dtype}")

    decoded = vae.decode(latent_samples)

    # Check for NaN in decoded output and attempt float32 retry
    if torch.isnan(decoded).any():
        print(f"[GridTester] ⚠️ VAE decode produced NaN! decoded shape={decoded.shape} dtype={decoded.dtype}")
        print(f"[GridTester] ⚠️ Input latent: shape={latent_samples.shape} dtype={latent_samples.dtype} device={latent_samples.device}")
        print(f"[GridTester] 🔄 Retrying VAE decode with float32 latent...")
        decoded = vae.decode(latent_samples.to(torch.float32))
        if torch.isnan(decoded).any():
            print(f"[GridTester] ❌ VAE decode still NaN after float32 retry")
        else:
            print(f"[GridTester] ✅ float32 retry succeeded")

    # Convert to PIL Image
    # .detach() is a defensive no-op under @torch.inference_mode() (tensors are
    # already non-grad). Kept for safety in case this function is ever called
    # without the decorator, but inference_mode is the primary defense — see
    # module docstring for why.
    img_np = decoded.detach().cpu().float().numpy()

    # Remove extra dimensions (handle shapes like (1, 1, H, W, C) or (1, H, W, C))
    while img_np.ndim > 3:
        img_np = img_np[0]

    # Now should be (H, W, C) or (C, H, W)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # Handle different channel orders
    if img_np.shape[0] == 3 and img_np.ndim == 3:  # CHW format
        img_np = np.transpose(img_np, (1, 2, 0))
    elif img_np.shape[-1] != 3 and img_np.ndim == 3:  # Not HWC format
        img_np = np.transpose(img_np, (1, 2, 0))

    return Image.fromarray(img_np)


def save_image_to_disk(image, output_dir, filename):
    """
    Save PIL Image to disk.
    
    Args:
        image: PIL.Image object
        output_dir: Directory to save in
        filename: Filename (including extension)
        
    Returns:
        str: Full path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    return filepath


def create_image_metadata(config, width, height, duration, seed, batch_idx, actual_positive_prompt, actual_negative_prompt, gen_index=None):
    """
    Create metadata dictionary for an image.

    Args:
        config: Configuration dictionary
        width: Image width
        height: Image height
        duration: Generation duration in seconds
        seed: Random seed used
        batch_idx: Batch index
        actual_positive_prompt: Final positive prompt (with triggers)
        actual_negative_prompt: Final negative prompt
        gen_index: Sequential generation index for deterministic sort ordering (optional, backwards-compatible)

    Returns:
        dict: Metadata dictionary
    """
    meta = config.copy()
    
    # Remove global settings that should only be in manifest.meta, not in individual items
    # These are session-wide settings that don't change per-image
    global_settings_to_remove = [
        "lora_triggerwords_append_settings",
        "lora_omit_triggers",
        "seed_behavior",
        "gguf_options",
        "model_prompt_prefix",
        "model_prompt_suffix"
    ]

    for key in global_settings_to_remove:
        meta.pop(key, None)

    # Remove attention_mode if it's "default" (keep manifest backward-compatible)
    if meta.get("attention_mode") == "default":
        meta.pop("attention_mode", None)

    update_dict = {
        "width": width,
        "height": height,
        "duration": duration,
        "seed": seed,
        "batch_idx": batch_idx,
        "positive": actual_positive_prompt,
        "negative": actual_negative_prompt
    }
    if gen_index is not None:
        update_dict["gen_index"] = gen_index
    # Preserve raw config prompts (without trigger words) for dashboard toggle
    meta["config_positive"] = config.get("positive", "")
    meta["config_negative"] = config.get("negative", "")
    meta.update(update_dict)

    return meta


def calculate_eta(job_durations, current_job, total_jobs):
    """
    Calculate ETA based on average job duration.
    
    Args:
        job_durations: List of previous job durations
        current_job: Current job number (1-indexed)
        total_jobs: Total number of jobs
        
    Returns:
        dict: Dictionary with eta info (hours, minutes, seconds, finish_time, finish_formatted)
    """
    if not job_durations:
        return None

    # Use rolling window of last 10 jobs for more responsive ETA
    # This adapts faster when upscaling patterns change (e.g., some configs have upscale, some don't)
    recent_window = job_durations[-10:] if len(job_durations) > 10 else job_durations
    avg_duration = sum(recent_window) / len(recent_window)
    remaining_jobs = total_jobs - current_job
    estimated_seconds = avg_duration * remaining_jobs
    
    eta_hours = int(estimated_seconds // 3600)
    eta_minutes = int((estimated_seconds % 3600) // 60)
    eta_seconds = int(estimated_seconds % 60)
    eta_finish_time = time.time() + estimated_seconds
    eta_finish_formatted = time.strftime("%H:%M:%S", time.localtime(eta_finish_time))
    
    return {
        "hours": eta_hours,
        "minutes": eta_minutes,
        "seconds": eta_seconds,
        "finish_time": eta_finish_time,
        "finish_formatted": eta_finish_formatted,
        "avg_duration": avg_duration
    }


def print_generation_progress(current_job, total_jobs, config, width, height, duration, eta_info):
    """
    Print progress information for current generation.
    
    Args:
        current_job: Current job number
        total_jobs: Total jobs
        config: Configuration dict
        width: Image width
        height: Image height
        duration: Generation duration
        eta_info: ETA info dict from calculate_eta()
    """
    progress_pct = int((current_job / total_jobs) * 100)
    
    print(f"{'='*80}")
    print(f"[GridTester] 📊 Job {current_job}/{total_jobs} ({progress_pct}%)")
    print(f"[GridTester] 🎨 {config['sampler']} @ {config['steps']} steps | {width}x{height}")
    print(f"[GridTester] ⏱️  {duration:.1f}s | Avg: {eta_info['avg_duration']:.1f}s/job")
    
    if eta_info['hours'] > 0:
        print(f"[GridTester] 🕒 ETA: {eta_info['hours']}h {eta_info['minutes']}m (finish ~{eta_info['finish_formatted']})")
    elif eta_info['minutes'] > 0:
        print(f"[GridTester] 🕒 ETA: {eta_info['minutes']}m {eta_info['seconds']}s (finish ~{eta_info['finish_formatted']})")
    else:
        print(f"[GridTester] 🕒 ETA: {eta_info['seconds']}s (finish ~{eta_info['finish_formatted']})")
    
    print(f"{'='*80}")


def flush_batch_with_vae(pending_batch, vae, img_dir, existing_data, session_name, manifest_path=None, unique_id=None, image_format="webp"):
    """
    Flush a batch of latents by decoding them with VAE and saving.

    Args:
        pending_batch: List of (latent_samples, metadata) tuples
        vae: VAE model for decoding
        img_dir: Image output directory
        existing_data: Manifest data dict
        session_name: Session name for filenames
        manifest_path: Path to manifest.json file (optional, enables disk syncing)
        unique_id: Node unique ID for dashboard updates (optional, enables dashboard updates)
        image_format: Image file format — "webp" (default), "png", or "jpg"

    Returns:
        int: Number of images saved
    """
    if not pending_batch:
        return 0

    import random

    saved_count = 0

    # Normalise format string
    fmt = (image_format or "webp").lower().strip()
    if fmt not in ("webp", "png", "jpg", "jpeg"):
        fmt = "webp"
    ext = "jpg" if fmt in ("jpg", "jpeg") else fmt

    for latent_samples, meta in pending_batch:
        # Decode latent to image
        image = decode_latent_with_vae(vae, latent_samples)

        # Generate ID using same format as remote_vae
        ts = int(time.time() * 100000) + random.randint(0, 1000)
        meta["id"] = ts

        # Generate filename with the chosen extension
        filename = f"img_{meta['id']}.{ext}"

        # Save image in chosen format
        filepath = os.path.join(img_dir, filename)
        if fmt == "png":
            image.save(filepath, format="PNG")
        elif fmt in ("jpg", "jpeg"):
            image.save(filepath, format="JPEG", quality=95)
        else:
            image.save(filepath, quality=80)
        
        # Normalize denoise to int if it's 1.0 (like remote_vae)
        if meta.get("denoise") == 1.0:
            meta["denoise"] = 1
        
        # Update meta with file path and rejected flag
        meta.update({
            "file": f"/view?filename={filename}&type=output&subfolder=benchmarks/{session_name}/images",
            "rejected": False
        })
        
        # Update manifest - insert at beginning (like remote_vae)
        existing_data["items"].insert(0, meta)
        
        # Sync with disk manifest to preserve tags (like remote_vae) - only if manifest_path provided
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    disk_manifest = json.load(f)
                
                # Create a lookup map for items currently in memory
                memory_items_map = {
                    i.get("id"): i 
                    for i in existing_data.get("items", []) 
                    if "id" in i
                }

                # Check every item on disk. If it exists in memory, copy the tags over.
                for disk_item in disk_manifest.get("items", []):
                    d_id = disk_item.get("id")
                    if d_id and d_id in memory_items_map:
                        local_item = memory_items_map[d_id]
                        
                        # PRESERVE TAGS: Copy these keys from disk to memory
                        if "favorited" in disk_item:
                            local_item["favorited"] = disk_item["favorited"]
                        if "rejected" in disk_item:
                            local_item["rejected"] = disk_item["rejected"]

            except Exception as e:
                print(f"[GridTester] ⚠️ Error syncing with disk manifest: {e}")

        # Save manifest to disk (like remote_vae) - only if manifest_path provided
        if manifest_path:
            with open(manifest_path, "w") as f:
                json.dump(existing_data, f, indent=4)
        
        # Send update to dashboard (like remote_vae) - only if unique_id provided
        if unique_id:
            try:
                from server import PromptServer
                if PromptServer:
                    # Get meta, use empty dict if not present
                    manifest_meta = existing_data.get("meta", {})
                    
                    PromptServer.instance.send_sync("ultimate_grid.update", {
                        "node": unique_id,
                        "session_name": session_name,
                        "new_items": [meta],
                        "meta": manifest_meta
                    })
            except (ImportError, KeyError):
                # Silently ignore dashboard update errors
                pass
        
        saved_count += 1
    
    print(f"[GridTester] 💾 Flushed {saved_count} images")
    return saved_count


def flush_batch_with_remote_vae(pending_batch, remote_vae_worker, existing_data, session_name):
    """
    Flush a batch of latents by sending them to remote VAE worker.
    
    Args:
        pending_batch: List of (latent_samples, metadata) tuples
        remote_vae_worker: RemoteVAEDecodeWorker instance
        existing_data: Manifest data dict (not used - worker handles manifest)
        session_name: Session name for filenames
        
    Returns:
        int: Number of images queued
    """
    if not pending_batch:
        return 0
    
    import random
    
    queued_count = 0
    
    for latent_samples, meta in pending_batch:
        # Generate ID using same format as old code
        ts = int(time.time() * 100000) + random.randint(0, 1000)
        meta["id"] = ts  # ← Set ID directly on meta dict
        
        # Queue the job - worker will create manifest entry
        # Pass the original meta dict, not a new one
        remote_vae_worker.add_job(
            latent_samples, 
            meta,  # ← Pass original meta, not a new dict
            meta["height"],
            meta["width"]
        )
        queued_count += 1
    
    print(f"[GridTester] 🌐 Queued {queued_count} images for remote VAE decoding")
    return queued_count


# =============================================================================
# SEEDVR2 UPSCALE — Calls ComfyUI-SeedVR2_VideoUpscaler nodes programmatically
# Requires ComfyUI-SeedVR2_VideoUpscaler to be installed as a dependency.
# =============================================================================

# @torch.inference_mode() — see module docstring. SeedVR2 is a diffusion-based
# upscaler with its own iterative sampling loop, identical autograd concerns
# to SDXL KSampler. The CurrentNodeContext below sets up V3-API execution but
# does NOT include inference_mode — that's our job at the function boundary.
@torch.inference_mode()
def seedvr2_upscale(pil_image, seedvr2_config):
    """
    Upscale an image using SeedVR2 diffusion-based upscaler.

    Args:
        pil_image: PIL Image (RGB)
        seedvr2_config: dict with SeedVR2 options (dit_model, resolution, seed, etc.)

    Returns:
        tuple: (pil_result, width, height, duration)
    """
    import time
    import torch
    import numpy as np

    t0 = time.time()
    sv = seedvr2_config

    # Find SeedVR2 nodes from ComfyUI's node registry.
    # SeedVR2 uses V3 API (ComfyExtension) so we scan NODE_CLASS_MAPPINGS directly
    # rather than trying to import the custom_nodes package.
    SeedVR2LoadDiTModel = None
    SeedVR2LoadVAEModel = None
    SeedVR2VideoUpscaler = None
    try:
        import nodes
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            if name == "SeedVR2LoadDiTModel":
                SeedVR2LoadDiTModel = cls
            elif name == "SeedVR2LoadVAEModel":
                SeedVR2LoadVAEModel = cls
            elif name == "SeedVR2VideoUpscaler":
                SeedVR2VideoUpscaler = cls
    except Exception:
        pass

    if not SeedVR2LoadDiTModel or not SeedVR2LoadVAEModel or not SeedVR2VideoUpscaler:
        raise RuntimeError(
            "SeedVR2 upscale requires ComfyUI-SeedVR2_VideoUpscaler to be installed.\n"
            "Install from: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler\n"
            "SeedVR2 nodes not found in ComfyUI's node registry."
        )

    print(f"[GridTester] 🎬 SeedVR2 upscale: model={sv.get('dit_model', '3b_fp8')}, "
          f"resolution={sv.get('resolution', 1080)}, seed={sv.get('seed', 42)}")

    # Set up V3 API execution context — SeedVR2 nodes require this
    # (they call get_executing_context().node_id internally)
    from comfy_execution.utils import CurrentNodeContext
    import uuid
    dummy_prompt_id = str(uuid.uuid4())

    # Convert PIL image to ComfyUI IMAGE tensor [1, H, W, 3] float32 [0,1]
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, 3]

    # All SeedVR2 execute() calls run inside a mock execution context
    # (V3 API nodes call get_executing_context().node_id internally)
    with CurrentNodeContext(prompt_id=dummy_prompt_id, node_id="uscg_seedvr2", list_index=0):
        # Load DiT model
        dit_result = SeedVR2LoadDiTModel.execute(
            model=sv.get("dit_model", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
            device="cuda:0",
            blocks_to_swap=sv.get("blocks_to_swap", 0),
            swap_io_components=False,
            offload_device=sv.get("offload_device", "cpu"),
            cache_model=sv.get("cache_model", False),
            attention_mode=sv.get("attention_mode", "sdpa")
        )
        dit = dit_result.output[0] if hasattr(dit_result, 'output') else dit_result[0]

        # Load VAE model
        vae_result = SeedVR2LoadVAEModel.execute(
            model="ema_vae_fp16.safetensors",
            device="cuda:0",
            encode_tiled=sv.get("encode_tiled", False),
            encode_tile_size=sv.get("encode_tile_size", 1024),
            encode_tile_overlap=sv.get("encode_tile_overlap", 128),
            decode_tiled=sv.get("decode_tiled", False),
            decode_tile_size=sv.get("decode_tile_size", 1024),
            decode_tile_overlap=sv.get("decode_tile_overlap", 128),
            offload_device=sv.get("vae_offload_device", "none"),
            cache_model=sv.get("vae_cache_model", False)
        )
        vae = vae_result.output[0] if hasattr(vae_result, 'output') else vae_result[0]

        # Run upscale
        upscale_result = SeedVR2VideoUpscaler.execute(
            image=img_tensor,
            dit=dit,
            vae=vae,
            seed=sv.get("seed", 42),
            resolution=sv.get("resolution", 1080),
            max_resolution=sv.get("max_resolution", 0),
            batch_size=sv.get("batch_size", 1),
            uniform_batch_size=False,
            color_correction=sv.get("color_correction", "lab"),
            input_noise_scale=sv.get("input_noise_scale", 0.0),
            latent_noise_scale=sv.get("latent_noise_scale", 0.0),
            offload_device=sv.get("offload_device", "cpu")
        )
        result_tensor = upscale_result.output[0] if hasattr(upscale_result, 'output') else upscale_result[0]

    # Convert back to PIL: [1, H', W', 3] float32 → PIL
    from PIL import Image
    result_np = (result_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    result_pil = Image.fromarray(result_np, "RGB")
    up_w, up_h = result_pil.size

    duration = round(time.time() - t0, 2)
    print(f"[GridTester] 🎬 SeedVR2 complete: {up_w}x{up_h} in {duration}s")

    return result_pil, up_w, up_h, duration