import re
import torch
import json
import os
import time
import random
import itertools
import folder_paths
import nodes
import comfy.utils
import comfy.sd
import comfy.samplers
from PIL import Image
import numpy as np
from .html_generator import get_html_template

class SamplerGridTester:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "positive_text": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl"}),
                "negative_text": ("STRING", {"multiline": True, "default": "bad quality, worst quality, lowres"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("STRING", {"default": "1.0", "multiline": False}), 
                "vae_batch_size": ("INT", {"default": 4, "min": -1, "max": 64}),
                "configs_json": ("STRING", {"multiline": True, "default": '[{"sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 7.0}]'}),
                "resolutions_json": ("STRING", {"default": '[[1024, 1024]]'}),
                "session_name": ("STRING", {"default": "my_session"}),
                "overwrite_existing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optional_model": ("MODEL",),
                "optional_clip": ("CLIP",),
                "optional_vae": ("VAE",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dashboard_html",)
    FUNCTION = "run_tests"
    CATEGORY = "sampling/testing"

    # --- HELPER: PARSE LORA STACKS ---
    def parse_lora_definition(self, lora_string, global_model_strength, global_clip_strength):
        if lora_string == "None": return []
        definitions = []
        parts = lora_string.split(" + ")
        for part in parts:
            part = part.strip()
            if ":" in part:
                segments = part.split(":")
                name = segments[0].strip()
                m_str = float(segments[1]) if len(segments) > 1 else 1.0
                c_str = float(segments[2]) if len(segments) > 2 else 1.0
                definitions.append((name, m_str, c_str))
            else:
                definitions.append((part, global_model_strength, global_clip_strength))
        return definitions

    # --- HELPER: PARSE DYNAMIC FLOAT ARRAYS ---
    def parse_float_input(self, input_str):
        try:
            val = json.loads(input_str)
            if isinstance(val, list): return [float(x) for x in val]
            return [float(val)]
        except:
            try:
                if "," in input_str: return [float(x.strip()) for x in input_str.split(",")]
                return [float(input_str)]
            except:
                return [1.0]

    # --- HELPER: PARSE STRING ARRAYS (PROMPTS) ---
    def parse_string_input(self, input_str):
        try:
            val = json.loads(input_str.strip())
            if isinstance(val, list):
                return [str(x) for x in val]
            return [str(val)]
        except:
            return [input_str]

    def run_tests(self, ckpt_name, positive_text, negative_text, seed, denoise, vae_batch_size, overwrite_existing, configs_json, resolutions_json, session_name, unique_id, optional_model=None, optional_clip=None, optional_vae=None, optional_positive=None, optional_negative=None, optional_latent=None):
        
        # --- ROBUST JSON ERROR HANDLING ---
        def parse_json_with_error(json_str, name):
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError as e:
                lines = json_str.split('\n')
                error_line_idx = e.lineno - 1
                bad_line = lines[error_line_idx]
                pointer = " " * (e.colno - 1) + "^"
                
                hint = ""
                if "Expecting property name" in e.msg:
                    preceding_text = json_str[:e.pos].rstrip()
                    if preceding_text.endswith(","):
                        hint = "\n>>> HINT: You have a TRAILING COMMA after the last item.\n>>> JSON does NOT allow commas before a closing bracket '}' or ']'.\n>>> Remove the comma on the line before this error."
                
                if "Expecting property name" in e.msg and "'" in bad_line:
                    hint += "\n>>> HINT: Ensure you are using DOUBLE QUOTES (\") for keys and strings, not single quotes (')."

                error_msg = (
                    f"\n\n[JSON ERROR] in {name}:\n"
                    f"---------------------------------------------------\n"
                    f"{e.msg}\n"
                    f"Line {e.lineno}:  {bad_line}\n"
                    f"          {pointer}\n"
                    f"---------------------------------------------------"
                    f"{hint}\n"
                )
                raise ValueError(error_msg)

        try:
            raw_configs = parse_json_with_error(configs_json, "Configs JSON")
            resolutions = parse_json_with_error(resolutions_json, "Resolutions JSON")
            denoise_values = self.parse_float_input(str(denoise))
            pos_prompts = self.parse_string_input(positive_text)
            neg_prompts = self.parse_string_input(negative_text)

        except Exception as e: 
            raise ValueError(f"{e}")

        # --- LOAD RESOURCES ---
        if optional_model is not None and optional_clip is not None:
            model = optional_model
            clip = optional_clip
            model_name_for_meta = "External Model"
        else:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae = out[:3]
            model_name_for_meta = ckpt_name

        if optional_vae is not None:
            vae = optional_vae
        elif 'vae' not in locals():
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            vae = out[2]

        session_name = re.sub(r'[^\w\-]', '', session_name)
        if not session_name: session_name = "default_session"
        
        base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)
        img_dir = os.path.join(base_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        manifest_path = os.path.join(base_dir, "manifest.json")

        existing_data = {"items": [], "meta": {}}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    d = json.load(f)
                    if isinstance(d, list): existing_data["items"] = d
                    else: existing_data = d
            except: pass

        existing_data["meta"] = {
            "model": model_name_for_meta,
            "positive": "Multiple" if len(pos_prompts) > 1 else pos_prompts[0],
            "negative": "Multiple" if len(neg_prompts) > 1 else neg_prompts[0],
            "updated": int(time.time())
        }

        # --- EXPAND CONFIGS ---
        ALL_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
        ALL_SAMPLERS = comfy.samplers.KSampler.SAMPLERS
        expanded = []
        for entry in raw_configs:
            def to_list(x): return x if isinstance(x, list) else [x]
            samplers = ALL_SAMPLERS if entry.get("sampler") == "*" else to_list(entry.get("sampler", "euler"))
            schedulers = ALL_SCHEDULERS if entry.get("scheduler") == "*" else to_list(entry.get("scheduler", "normal"))
            steps_l = to_list(entry.get("steps", 20))
            cfgs = to_list(entry.get("cfg", 7.0))
            loras = to_list(entry.get("lora", "None"))
            str_m = to_list(entry.get("str_model", 1.0))
            str_c = to_list(entry.get("str_clip", 1.0))
            
            for combo in itertools.product(samplers, schedulers, steps_l, cfgs, loras, str_m, str_c, denoise_values, pos_prompts, neg_prompts):
                expanded.append({
                    "sampler": combo[0], "scheduler": combo[1], "steps": combo[2],
                    "cfg": combo[3], "lora": combo[4], "str_model": combo[5], "str_clip": combo[6],
                    "denoise": combo[7],
                    "positive": combo[8],
                    "negative": combo[9]
                })

        # --- INTELLIGENT SORTING (OPTIMIZATION) ---
        # Sort by: 1. LoRA (Expensive), 2. Prompt (Moderate), 3. Sampler/Settings (Cheap)
        # This groups similar runs together to minimize model patching and text encoding.
        expanded.sort(key=lambda x: (x['lora'], x['str_model'], x['str_clip'], x['positive'], x['negative']))

        print(f"[GridTester] Processing {len(expanded) * len(resolutions)} items...")

        cached_lora_key = None
        cached_model, cached_clip = None, None
        
        # Condition Cache
        cached_pos_key, cached_neg_key = None, None
        cached_positive, cached_negative = None, None

        pending_batch = []
        MATCH_KEYS = ["sampler", "scheduler", "steps", "cfg", "lora", "str_model", "str_clip", "denoise", "seed", "width", "height", "positive", "negative"]

        def flush_batch(batch_list):
            if not batch_list: return
            latents_to_decode = torch.cat([x[0] for x in batch_list], dim=0)
            decoded = vae.decode(latents_to_decode)
            for i, img_tensor in enumerate(decoded):
                img_np = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                meta = batch_list[i][1]
                ts = int(time.time() * 100000) + random.randint(0,1000)
                filename = f"img_{ts}.webp"
                img.save(os.path.join(img_dir, filename), quality=80)
                meta.update({
                    "id": ts, 
                    "file": f"/view?filename={filename}&type=output&subfolder=benchmarks/{session_name}/images",
                    "rejected": False
                })
                existing_data["items"].insert(0, meta)
        
        for res in resolutions:
            w, h = res[0], res[1]
            flush_batch(pending_batch)
            pending_batch = []

            for conf in expanded:
                # --- CHECK FOR EXISTING MATCH ---
                match_index = -1
                for idx, item in enumerate(existing_data["items"]):
                    is_match = True
                    for k in MATCH_KEYS:
                        val_conf = conf.get(k)
                        if k == "width": val_conf = w
                        if k == "height": val_conf = h
                        if k == "seed": val_conf = seed
                        
                        if item.get(k) != val_conf:
                            is_match = False
                            break
                    if is_match:
                        match_index = idx
                        break

                if match_index != -1:
                    if not overwrite_existing:
                        continue
                    else:
                        old_item = existing_data["items"][match_index]
                        try:
                            old_fname_match = re.search(r'filename=([^&]+)', old_item["file"])
                            if old_fname_match:
                                old_fname = old_fname_match.group(1)
                                old_path = os.path.join(img_dir, old_fname)
                                if os.path.exists(old_path):
                                    os.remove(old_path)
                        except Exception as e:
                            print(f"Error cleaning up old file: {e}")
                        existing_data["items"].pop(match_index)

                # --- 1. OPTIMIZED MODEL LOADING ---
                active_loras = self.parse_lora_definition(conf["lora"], conf["str_model"], conf["str_clip"])
                current_lora_key = tuple(active_loras)
                
                # Only patch model if LoRA config changed
                if current_lora_key == cached_lora_key and cached_model is not None:
                    curr_model, curr_clip = cached_model, cached_clip
                else:
                    curr_model, curr_clip = model, clip
                    for lora_def in active_loras:
                        lname, lstr_m, lstr_c = lora_def
                        path = folder_paths.get_full_path("loras", lname)
                        if path:
                            lora_data = comfy.utils.load_torch_file(path)
                            curr_model, curr_clip = comfy.sd.load_lora_for_models(curr_model, curr_clip, lora_data, lstr_m, lstr_c)
                    cached_lora_key = current_lora_key
                    cached_model, cached_clip = curr_model, curr_clip
                    # Reset conditioning cache because Clip model changed
                    cached_positive, cached_negative = None, None 

                # --- 2. OPTIMIZED CONDITIONING ---
                if optional_positive is not None:
                    positive = optional_positive
                else:
                    # Check cache
                    if conf["positive"] == cached_pos_key and cached_positive is not None:
                        positive = cached_positive
                    else:
                        tokens = curr_clip.tokenize(conf["positive"])
                        cond, pooled = curr_clip.encode_from_tokens(tokens, return_pooled=True)
                        positive = [[cond, {"pooled_output": pooled}]]
                        cached_positive = positive
                        cached_pos_key = conf["positive"]

                if optional_negative is not None:
                    negative = optional_negative
                else:
                    # Check cache
                    if conf["negative"] == cached_neg_key and cached_negative is not None:
                        negative = cached_negative
                    else:
                        tokens = curr_clip.tokenize(conf["negative"])
                        cond, pooled = curr_clip.encode_from_tokens(tokens, return_pooled=True)
                        negative = [[cond, {"pooled_output": pooled}]]
                        cached_negative = negative
                        cached_neg_key = conf["negative"]

                # --- LATENT SETUP ---
                if optional_latent is not None:
                    latent_in = {"samples": optional_latent["samples"].clone()}
                else:
                    latent_in = {"samples": torch.zeros([1, 4, h // 8, w // 8])}

                # Generate
                t0 = time.time()
                try:
                    result = nodes.common_ksampler(
                        model=curr_model, seed=seed, steps=conf["steps"], cfg=conf["cfg"],
                        sampler_name=conf["sampler"], scheduler=conf["scheduler"],
                        positive=positive, negative=negative, latent=latent_in, 
                        denoise=conf["denoise"]
                    )
                except Exception as e:
                    print(f"Generation Failed: {e}")
                    continue
                
                duration = round(time.time() - t0, 3)
                meta = conf.copy()
                meta.update({
                    "width": w, "height": h, "duration": duration, "seed": seed
                })
                pending_batch.append((result[0]["samples"], meta))
                
                if vae_batch_size != -1 and len(pending_batch) >= vae_batch_size:
                    flush_batch(pending_batch)
                    pending_batch = []

        flush_batch(pending_batch)
        with open(manifest_path, "w") as f: json.dump(existing_data, f, indent=4)
        html = get_html_template(session_name, existing_data, unique_id)
        return (html,)