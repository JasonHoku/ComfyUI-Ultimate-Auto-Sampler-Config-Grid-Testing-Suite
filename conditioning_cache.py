"""
Conditioning Cache System for SamplerGridTester
Saves and loads encoded CLIP conditioning to avoid re-encoding prompts
"""

import os
import json
import torch
import hashlib
import base64
import numpy as np
from typing import Dict, Tuple, Optional, Any


class ConditioningCache:
    """
    Manages persistent caching of encoded conditioning tensors.
    
    The cache stores conditioning tensors as base64-encoded numpy arrays in JSON,
    keyed by a hash of the prompt text + CLIP model state.
    """
    
    def __init__(self, cache_dir: str, clip_hash: str = "unknown", enable_disk_cache: bool = True):
        """
        Initialize the conditioning cache.
        
        Args:
            cache_dir: Directory to store cache files (e.g., benchmarks/session_name/)
            clip_hash: Hash of the CLIP model to ensure compatibility
            enable_disk_cache: Whether to save/load cache to/from disk
        """
        self.cache_dir = cache_dir
        self.clip_hash = clip_hash
        self.cache_file = os.path.join(cache_dir, "conditioning_cache.json")
        self.enable_disk_cache = enable_disk_cache
        
        # Track current LoRA configuration for cache key generation
        self.current_lora_config = ""
        
        # Statistics - MUST be initialized BEFORE _load_from_disk()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "loads": 0
        }
        
        # In-memory cache for current session
        self.memory_cache = {
            "positive": {},
            "negative": {}
        }
        
        # Load existing cache from disk (uses self.stats) - only if enabled
        if self.enable_disk_cache:
            self.disk_cache = self._load_from_disk()
        else:
            self.disk_cache = {
                "version": "1.0",
                "clip_hash": self.clip_hash,
                "entries": {}
            }
            print(f"[CondCache] ℹ️ Disk cache disabled - using memory-only cache for this session")
    
    def _get_clip_hash(self, clip_model) -> str:
        """
        Generate a hash of the CLIP model state to ensure cache validity.
        
        Args:
            clip_model: The CLIP model object
            
        Returns:
            MD5 hash string of the model
        """
        try:
            # Try to hash the model's state dict
            if hasattr(clip_model, 'state_dict'):
                state_dict = clip_model.state_dict()
                # Just hash the keys and shapes, not full tensors (too slow)
                model_signature = str([(k, tuple(v.shape)) for k, v in list(state_dict.items())[:10]])
            elif hasattr(clip_model, 'cond_stage_model'):
                # Try alternative attribute
                model_signature = str(type(clip_model.cond_stage_model))
            else:
                model_signature = str(type(clip_model))
            
            return hashlib.md5(model_signature.encode()).hexdigest()[:16]
        except Exception as e:
            print(f"[CondCache] Warning: Could not hash CLIP model: {e}")
            return "unknown"
    
    def _prompt_key(self, prompt_text: str) -> str:
        """
        Generate a cache key from prompt text + LoRA configuration.
        
        Args:
            prompt_text: The prompt string
            
        Returns:
            MD5 hash of the prompt + CLIP hash + LoRA config
        """
        # Include LoRA configuration in the cache key to avoid incorrect cache hits
        # when using the same prompt with different LoRAs or strengths
        combined = f"{self.clip_hash}:{self.current_lora_config}:{prompt_text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def set_lora_config(self, lora_config: str):
        """
        Set the current LoRA configuration for cache key generation.
        
        This must be called whenever the LoRA configuration changes to ensure
        that cached conditioning is only reused when the CLIP model is in the
        same state (same LoRAs with same strengths).
        
        Args:
            lora_config: String describing the LoRA configuration, e.g.:
                         "None" or "lora1.safetensors:0.8:0.6,lora2.safetensors:1.0:1.0"
        """
        self.current_lora_config = lora_config
    
    def _conditioning_to_dict(self, conditioning) -> Dict[str, Any]:
        """
        Convert conditioning tensor to JSON-serializable format.
        
        Args:
            conditioning: ComfyUI conditioning format [[tensor, {pooled_output: tensor}]]
            
        Returns:
            Dictionary with base64-encoded tensors
        """
        try:
            cond_tensor = conditioning[0][0]  # Main conditioning tensor
            pooled_tensor = conditioning[0][1].get("pooled_output")
            
            # Convert to numpy and then to base64
            cond_np = cond_tensor.cpu().numpy()
            cond_b64 = base64.b64encode(cond_np.tobytes()).decode('utf-8')
            cond_shape = list(cond_np.shape)
            cond_dtype = str(cond_np.dtype)
            
            result = {
                "cond": {
                    "data": cond_b64,
                    "shape": cond_shape,
                    "dtype": cond_dtype
                }
            }
            
            if pooled_tensor is not None:
                pooled_np = pooled_tensor.cpu().numpy()
                pooled_b64 = base64.b64encode(pooled_np.tobytes()).decode('utf-8')
                pooled_shape = list(pooled_np.shape)
                pooled_dtype = str(pooled_np.dtype)
                
                result["pooled"] = {
                    "data": pooled_b64,
                    "shape": pooled_shape,
                    "dtype": pooled_dtype
                }
            
            return result
        except Exception as e:
            print(f"[CondCache] Error serializing conditioning: {e}")
            return None
    
    def _dict_to_conditioning(self, data: Dict[str, Any]):
        """
        Convert JSON dict back to conditioning tensor.
        
        Args:
            data: Dictionary with base64-encoded tensors
            
        Returns:
            ComfyUI conditioning format [[tensor, {pooled_output: tensor}]]
        """
        try:
            # Reconstruct main conditioning tensor
            cond_b64 = data["cond"]["data"]
            cond_shape = tuple(data["cond"]["shape"])
            cond_dtype = np.dtype(data["cond"]["dtype"])
            
            cond_bytes = base64.b64decode(cond_b64)
            cond_np = np.frombuffer(cond_bytes, dtype=cond_dtype).reshape(cond_shape)
            cond_tensor = torch.from_numpy(cond_np)
            
            # Reconstruct pooled output if present
            pooled_tensor = None
            if "pooled" in data:
                pooled_b64 = data["pooled"]["data"]
                pooled_shape = tuple(data["pooled"]["shape"])
                pooled_dtype = np.dtype(data["pooled"]["dtype"])
                
                pooled_bytes = base64.b64decode(pooled_b64)
                pooled_np = np.frombuffer(pooled_bytes, dtype=pooled_dtype).reshape(pooled_shape)
                pooled_tensor = torch.from_numpy(pooled_np)
            
            # Return in ComfyUI format
            return [[cond_tensor, {"pooled_output": pooled_tensor}]]
        except Exception as e:
            print(f"[CondCache] Error deserializing conditioning: {e}")
            return None
    
    @staticmethod
    def _tensor_to_json(t) -> Dict[str, Any]:
        arr = np.ascontiguousarray(t.detach().cpu().numpy())
        return {"t": "tensor", "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
                "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def _conditioning_to_entries(self, conditioning) -> Optional[list]:
        """
        Serialize EVERY conditioning entry and EVERY extra key (AND prompts, attention masks,
        regional tags...). Tensors become base64 numpy, None stays None, anything else must
        survive json.dumps. Returns None when something cannot be stored (the entry then stays
        memory-only instead of being silently truncated).
        """
        try:
            entries = []
            for item in conditioning:
                tensor = item[0]
                extras = item[1] if len(item) > 1 else {}
                if not hasattr(tensor, "detach"):
                    return None
                ser_extras = {}
                for k, v in (extras or {}).items():
                    if v is None:
                        ser_extras[k] = {"t": "none"}
                    elif hasattr(v, "detach"):
                        ser_extras[k] = self._tensor_to_json(v)
                    else:
                        json.dumps(v)  # raises for anything not JSON-safe
                        ser_extras[k] = {"t": "value", "v": v}
                entries.append({"cond": self._tensor_to_json(tensor), "extras": ser_extras})
            return entries
        except Exception as e:
            print(f"[CondCache] Not caching to disk (unserializable conditioning): {e}")
            return None

    @staticmethod
    def _tensor_from_json(d):
        raw = base64.b64decode(d["data"])
        arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"])).copy()
        return torch.from_numpy(arr)

    def _entries_to_conditioning(self, entries):
        """Inverse of _conditioning_to_entries. Returns None on any problem (treated as a cache miss)."""
        try:
            result = []
            for e in entries:
                extras = {}
                for k, v in e.get("extras", {}).items():
                    kind = v.get("t")
                    if kind == "none":
                        extras[k] = None
                    elif kind == "tensor":
                        extras[k] = self._tensor_from_json(v)
                    elif kind == "value":
                        extras[k] = v["v"]
                    else:
                        return None
                result.append([self._tensor_from_json(e["cond"]), extras])
            return result or None
        except Exception as ex:
            print(f"[CondCache] Error deserializing conditioning entries: {ex}")
            return None

    def _load_from_disk(self) -> Dict[str, Any]:
        """
        Load cache from disk.
        
        Returns:
            Dictionary containing cached conditioning data
        """
        if not os.path.exists(self.cache_file):
            return {
                "version": "1.0",
                "clip_hash": self.clip_hash,
                "entries": {}
            }
        
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            
            # Validate CLIP hash matches
            if data.get("clip_hash") != self.clip_hash:
                print(f"[CondCache] ⚠️ CLIP model changed (cache: {data.get('clip_hash')}, current: {self.clip_hash})")
                print(f"[CondCache] 🔄 Invalidating old cache...")
                return {
                    "version": "1.0",
                    "clip_hash": self.clip_hash,
                    "entries": {}
                }
            
            num_entries = len(data.get("entries", {}))
            print(f"[CondCache] 📂 Loaded {num_entries} cached conditioning entries")
            self.stats["loads"] = num_entries
            
            # Debug: Show a sample of what's in the cache
            if num_entries > 0:
                sample_keys = list(data.get("entries", {}).keys())[:3]
                print(f"[CondCache] 🔍 Sample cache keys: {sample_keys}")
            
            return data
        except Exception as e:
            print(f"[CondCache] ⚠️ Error loading cache: {e}")
            import traceback
            traceback.print_exc()
            return {
                "version": "1.0",
                "clip_hash": self.clip_hash,
                "entries": {}
            }
    
    def _save_to_disk(self):
        """Save cache to disk."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            with open(self.cache_file, "w") as f:
                json.dump(self.disk_cache, f, indent=2)
            
            num_entries = len(self.disk_cache.get("entries", {}))
            print(f"[CondCache] 💾 Saved {num_entries} conditioning entries to cache")
        except Exception as e:
            print(f"[CondCache] ⚠️ Error saving cache: {e}")
    
    def get(self, prompt_text: str, prompt_type: str = "positive", debug: bool = False) -> Optional[Any]:
        """
        Get cached conditioning for a prompt.
        
        Args:
            prompt_text: The prompt string
            prompt_type: "positive" or "negative"
            debug: Enable debug logging
            
        Returns:
            Conditioning tensor or None if not cached
        """
        # Check memory cache first (uses full text as key)
        if prompt_text in self.memory_cache[prompt_type]:
            self.stats["hits"] += 1
            if debug:
                print(f"[CondCache] ✅ Memory cache hit for: {prompt_text[:50]}...")
            return self.memory_cache[prompt_type][prompt_text]
        
        # Check disk cache (uses hash as key) - only if enabled
        if not self.enable_disk_cache:
            self.stats["misses"] += 1
            if debug:
                print(f"[CondCache] ❌ Cache miss (disk cache disabled)")
            return None
        
        key = self._prompt_key(prompt_text)
        if debug:
            print(f"[CondCache] 🔍 Looking for key: {key}")
            print(f"[CondCache] 🔍 Prompt: {prompt_text[:80]}...")
        
        if key in self.disk_cache.get("entries", {}):
            entry = self.disk_cache["entries"][key]
            # conditioning_v2 keeps every entry + extra key. Legacy entries (single-entry
            # "conditioning" only) dropped AND prompts / regional tags, so they count as a miss.
            conditioning = None
            if "conditioning_v2" in entry:
                conditioning = self._entries_to_conditioning(entry["conditioning_v2"])
            
            if conditioning is not None:
                # Store in memory cache for fast access (using full text)
                self.memory_cache[prompt_type][prompt_text] = conditioning
                self.stats["hits"] += 1
                if debug:
                    print(f"[CondCache] ✅ Disk cache hit!")
                return conditioning
        
        self.stats["misses"] += 1
        if debug:
            print(f"[CondCache] ❌ Cache miss")
        return None
    
    def set(self, prompt_text: str, conditioning, prompt_type: str = "positive"):
        """
        Cache conditioning for a prompt.
        
        Args:
            prompt_text: The prompt string
            conditioning: The conditioning tensor
            prompt_type: "positive" or "negative"
        """
        # Store in memory cache (always enabled)
        self.memory_cache[prompt_type][prompt_text] = conditioning
        
        # Store in disk cache (only if enabled)
        if not self.enable_disk_cache:
            return
        
        key = self._prompt_key(prompt_text)
        serialized = self._conditioning_to_entries(conditioning)
        
        if serialized is not None:
            if "entries" not in self.disk_cache:
                self.disk_cache["entries"] = {}
            
            self.disk_cache["entries"][key] = {
                "prompt": prompt_text[:200],  # Truncate for readability
                "type": prompt_type,
                "conditioning_v2": serialized
            }
            
            self.stats["saves"] += 1
    
    def save(self):
        """Save the cache to disk (only if disk cache is enabled)."""
        if self.enable_disk_cache:
            self._save_to_disk()
        else:
            print(f"[CondCache] ℹ️ Skipping disk save (disk cache disabled)")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        hit_rate = 0
        if (self.stats["hits"] + self.stats["misses"]) > 0:
            hit_rate = (self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])) * 100
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 1),
            "total_cached": len(self.disk_cache.get("entries", {}))
        }
    
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"\n{'='*80}")
        print(f"[CondCache] 📊 CONDITIONING CACHE STATISTICS")
        print(f"{'='*80}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']}%")
        print(f"  Saved: {stats['saves']}")
        print(f"  Total Cached: {stats['total_cached']}")
        print(f"{'='*80}\n")