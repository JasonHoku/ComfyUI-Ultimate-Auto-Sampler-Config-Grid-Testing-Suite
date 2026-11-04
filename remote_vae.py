import torch
import hashlib
import queue
import os
import threading
import time
import numpy as np
from PIL import Image

from . import image_metrics
try:
    from diffusers.image_processor import VaeImageProcessor
except ImportError as e:
    VaeImageProcessor = None
    print(f"[GridTester] ⚠️ Could not import diffusers: {e}")
    print("[GridTester] ⚠️ Remote VAE features will be unavailable.")
    print("[GridTester] ⚠️ To fix, run: pip install --upgrade diffusers>=0.25.0")
    print("[GridTester] ⚠️ If another custom node requires an older version, you may need to resolve the conflict manually.")



INSTALL_INSTRUCTIONS = (
    "Remote VAE requires the ComfyUI-USCG-RemoteVAE companion plugin.\n"
    "Install via Comfy Manager (search 'USCG Remote VAE') or:\n"
    "  git clone https://github.com/JasonHoku/ComfyUI-USCG-RemoteVAE\n"
    "into your ComfyUI/custom_nodes/ directory."
)


def is_remote_vae_available():
    """Return True if the ComfyUI-USCG-RemoteVAE companion plugin is loaded."""
    try:
        import comfyui_uscg_remote_vae  # noqa: F401
        return True
    except ImportError:
        return False


def get_endpoint_names():
    """Return list of HF endpoint names for UI dropdown.
    Empty list if companion not installed."""
    try:
        from comfyui_uscg_remote_vae import list_endpoints
        return [name for name, _url in list_endpoints()]
    except ImportError:
        return []


def _companion_decode(endpoint, tensor, height, width):
    """Forward a decode call to the companion plugin.

    Raises:
        RuntimeError: companion not installed (with install instructions)
                      or companion version too old.
    """
    try:
        from comfyui_uscg_remote_vae import decode, __version__
    except ImportError:
        raise RuntimeError(INSTALL_INSTRUCTIONS)
    if tuple(map(int, __version__.split(".")[:2])) < (0, 1):
        raise RuntimeError(
            f"ComfyUI-USCG-RemoteVAE >= 0.1.0 required (found {__version__}). "
            "Update via Comfy Manager."
        )
    return decode(endpoint, tensor, height, width)


def detect_model_type(model, latent_channels):
    """
    Auto-detect model type for HF Remote VAE endpoint selection
    """
    #print(f"[GridTester] 🔍 Detecting model type... latent_channels={latent_channels}")
    
    if latent_channels == 16:
        # Could be Flux, SD3, or HunyuanVideo
        if hasattr(model, 'model'):
            model_str = str(type(model.model)).lower()
            if 'xl' in model_str:
                return "SDXL"
        if hasattr(model, 'model') and hasattr(model.model, 'model_type'):
            model_type = str(model.model.model_type).lower()
            #print(f"[GridTester] 🔍 Model type attribute: {model_type}")
            if 'flux' in model_type:
                return "Flux"
            elif 'hunyuan' in model_type or 'video' in model_type:
                return "HunyuanVideo"
        
        # Check model name/path
        if hasattr(model, 'model') and hasattr(model.model, 'model_config'):
            config = str(model.model.model_config).lower()
            if 'flux' in config:
                return "Flux"
        
        # Default to Flux for 16-channel
        #print(f"[GridTester] 🔍 Defaulting to Flux for 16 channels")
        return "Flux"
    
    elif latent_channels == 4:
        # SD or SDXL - check resolution or model size
        # 1. Check model name FIRST (lightweight) ✅
        if hasattr(model, 'model'):
            model_str = str(type(model.model)).lower()
            if 'xl' in model_str or 'sdxl' in model_str:
                return "SDXL"
        
        # 2. Check model config (lightweight) ✅
        if hasattr(model, 'model') and hasattr(model.model, 'model_config'):
            config_str = str(model.model.model_config).lower()
            if 'sdxl' in config_str:
                return "SDXL"
        
        # 3. ONLY count parameters as LAST RESORT ✅
        with torch.no_grad():
            try:
                param_count = sum(p.numel() for p in model.model.diffusion_model.parameters())
                #print(f"[GridTester] 🔍 Model parameter count: {param_count:,}")
                
                if param_count > 1_000_000_000:  # > 1B params suggests SDXL
                    #print(f"[GridTester] 🔍 Detected SDXL (>1B params)")
                    return "SDXL"
                else:
                    #print(f"[GridTester] 🔍 Detected SD1.5 (<1B params)")
                    return "SD"
            except:
                pass
        
        # Fallback: check if model name contains 'xl'
        if hasattr(model, 'model'):
            model_str = str(type(model.model)).lower()
            if 'xl' in model_str or 'sdxl' in model_str:
                #print(f"[GridTester] 🔍 Detected SDXL from model name")
                return "SDXL"
        
        #print(f"[GridTester] 🔍 Defaulting to SD for 4 channels")
        return "SD"
    
    else:
        #print(f"[GridTester] ⚠️ Unknown latent channels: {latent_channels}, defaulting to SD")
        return "SD"


class RemoteVAEDecodeWorker:
    """
    Background worker thread for async remote VAE decoding
    """
    def __init__(self, endpoint, img_dir, manifest_path, existing_data, session_name, unique_id, image_format="webp"):
        self.endpoint = endpoint
        self.img_dir = img_dir
        self.manifest_path = manifest_path
        self.existing_data = existing_data
        self.session_name = session_name
        self.unique_id = unique_id
        fmt = (image_format or "webp").lower().strip()
        if fmt not in ("webp", "png", "jpg", "jpeg"):
            fmt = "webp"
        self.image_format = fmt
        # Captured now so a late worker thread is unaffected by end_run() ordering
        self._metrics_spec = image_metrics.current_spec()
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.total_decoded = 0
    
    def _worker(self):
        """Worker thread that processes decode requests"""
        while True:
            item = self.queue.get()
            
            if item is None:  # Poison pill to stop thread
                self.queue.task_done()
                break
            
            try:
                latent_tensor, meta, height, width = item
                
                # print(f"[GridTester] 🌐 Processing remote decode for image #{meta['id']}")
                # print(f"[GridTester] 🌐 Input latent shape: {latent_tensor.shape}")
                
                # Ensure batch dimension exists
                if latent_tensor.ndim == 3:
                    latent_tensor = latent_tensor.unsqueeze(0)
                    print(f"[GridTester] 🌐 Added batch dim: {latent_tensor.shape}")
                
                # Remote decode - returns [B, C, H, W] tensor
                decoded = _companion_decode(self.endpoint, latent_tensor, height, width)
                
                # print(f"[GridTester] 🌐 Decoded shape: {decoded.shape}")
                # print(f"[GridTester] 🌐 Decoded dtype: {decoded.dtype}")
                
                # Use VaeImageProcessor to properly postprocess the VAE output
                # This handles denormalization from [-1, 1] to [0, 1] and format conversion
                if VaeImageProcessor is None:
                    raise ImportError("diffusers is not installed or outdated. Fix: pip install --upgrade diffusers")
                image_processor = VaeImageProcessor(vae_scale_factor=8)
                image_processor.config.do_resize = False
                
                # Get the result and check what format it's in
                result = image_processor.postprocess(decoded, output_type="pt")
                #print(f"[GridTester] 🔧 Postprocessed shape: {result.shape}")
                #print(f"[GridTester] 🔧 Postprocessed dtype: {result.dtype}")
                
                # The postprocess with output_type="pt" returns [B, C, H, W] in [0, 1]
                # We need to convert to [H, W, C] for PIL
                image = result[0]  # Remove batch: [C, H, W]
                image = image.permute(1, 2, 0)  # Convert to [H, W, C]
                image = image.cpu().numpy()
                image_np = (image * 255).round().astype(np.uint8)
                
                #print(f"[GridTester] 🔧 Final image shape: {image_np.shape}, dtype: {image_np.dtype}")
                
                # Create PIL Image
                img = Image.fromarray(image_np)
                
                # Save image in chosen format
                fmt = self.image_format  # Already normalised in __init__
                ext = "jpg" if fmt in ("jpg", "jpeg") else fmt
                filename = f"img_{meta['id']}.{ext}"
                filepath = os.path.join(self.img_dir, filename)
                if fmt == "png":
                    img.save(filepath, format="PNG")
                elif fmt in ("jpg", "jpeg"):
                    img.save(filepath, format="JPEG", quality=95)
                else:
                    img.save(filepath, quality=80)

                meta.update({
                    "file": f"/view?filename={filename}&type=output&subfolder=benchmarks/{self.session_name}/images",
                    "rejected": False
                })
                
                # Optional image metrics from the postprocessed [B,C,H,W] 0-1 tensor
                image_metrics.attach(meta, result[:1], spec=self._metrics_spec)

                # Update manifest
                import json
                self.existing_data["items"].insert(0, meta)
                if os.path.exists(self.manifest_path):
                    try:
                        with open(self.manifest_path, "r") as f:
                            disk_manifest = json.load(f)
                        
                        # Create a lookup map for items currently in memory
                        # This allows us to update self.existing_data in-place
                        memory_items_map = {
                            i.get("id"): i 
                            for i in self.existing_data.get("items", []) 
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

                # 3. Now it is safe to save (Memory is now 'fresh' with disk changes)
                with open(self.manifest_path, "w") as f:
                    json.dump(self.existing_data, f, indent=4)
                
                # print("Save Manifest at remote vae")
                # Send update to dashboard
                try:
                    from server import PromptServer
                    if PromptServer:
                        # Get meta, use empty dict if not present
                        manifest_meta = self.existing_data.get("meta", {})
                        
                        PromptServer.instance.send_sync("ultimate_grid.update", {
                            "node": self.unique_id,
                            "session_name": self.session_name,
                            "new_items": [meta],
                            "meta": manifest_meta
                        })
                except (ImportError, KeyError):
                    # Silently ignore dashboard update errors
                    pass
                
                self.total_decoded += 1
                # print(f"[GridTester] ✅ Remote VAE decoded #{meta['id']} ({self.total_decoded} total)")
                
            except Exception as e:
                print(f"[GridTester] ❌ Remote VAE worker error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.queue.task_done()

    
    def add_job(self, latent_tensor, meta, height, width):
        """Add a decode job to the queue"""
        self.queue.put((latent_tensor, meta, height, width))
    
    def wait_completion(self):
        """Wait for all jobs to complete"""
        self.queue.join()
    
    def stop(self):
        """Stop the worker thread"""
        self.queue.put(None)
        self.thread.join()