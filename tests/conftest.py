"""
Pytest configuration for standalone testing outside ComfyUI.

The custom node's __init__.py requires ComfyUI internals (server, folder_paths,
torch, etc.) that are not available in a plain Python environment.
We stub those out here so pytest can collect and run unit tests without ComfyUI.
"""
import sys
import types
import os

# Try to import real torch first — some tests (crop/paste, integration) need it.
# If torch isn't installed in the active Python, fall through to the stub loop below.
try:
    import torch  # noqa: F401 — surfacing real torch into sys.modules
except ImportError:
    pass

# ── Stub out ComfyUI-only modules before any test or __init__.py import ──────
_COMFY_STUBS = [
    "server",
    "folder_paths",
    "nodes",
    "torch",
    "aiohttp",
    "aiohttp.web",
    "comfy",
    "comfy.utils",
    "comfy.sd",
    "comfy.model_management",
    "comfy.samplers",
    # diffusers triggers a torchvision.__spec__ is None ValueError when imported
    # outside ComfyUI.  Stub it out so remote_vae.py's try/except catches the
    # ImportError and sets VaeImageProcessor = None.
    "diffusers",
    "diffusers.image_processor",
]

for _mod_name in _COMFY_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        # aiohttp.web needs a minimal RouteTableDef stub used by distribution_routes
        if _mod_name == "aiohttp.web":
            class _RouteTableDef:  # noqa: N801
                def get(self, *a, **kw):
                    return lambda fn: fn
                def post(self, *a, **kw):
                    return lambda fn: fn
                def delete(self, *a, **kw):
                    return lambda fn: fn
                def put(self, *a, **kw):
                    return lambda fn: fn
            _stub.RouteTableDef = _RouteTableDef
        # comfy.samplers needs KSampler with SCHEDULERS and SAMPLERS lists
        if _mod_name == "comfy.samplers":
            class _KSampler:  # noqa: N801
                SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
                SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                            "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc",
                            "uni_pc_bh2"]
            _stub.KSampler = _KSampler
        sys.modules[_mod_name] = _stub

# Wire comfy.samplers as an attribute of the comfy stub so
# `import comfy.samplers` and `comfy.samplers.KSampler` both resolve
if hasattr(sys.modules.get("comfy"), "__name__"):
    sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]

# comfy.model_management needs InterruptProcessingException for generation_orchestrator
if "comfy.model_management" in sys.modules:
    if not hasattr(sys.modules["comfy.model_management"], "InterruptProcessingException"):
        sys.modules["comfy.model_management"].InterruptProcessingException = type(
            "InterruptProcessingException", (Exception,), {}
        )

# Ensure the custom node root is on sys.path so `ltx_video_generation` is importable
_NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _NODE_ROOT not in sys.path:
    sys.path.insert(0, _NODE_ROOT)

# ── Package stub so generation_orchestrator's relative imports resolve ────────
# generation_orchestrator.py uses `from .trigger_words import ...` etc.
# When imported as a top-level module (sys.path.insert approach), relative
# imports fail unless we set up a minimal package context in sys.modules.
_PKG_NAME = os.path.basename(_NODE_ROOT)
if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [_NODE_ROOT]
    _pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = _pkg

# Stub the sub-modules that generation_orchestrator imports via relative paths
_GEN_ORCH_DEPS = {
    "trigger_words": {
        "collect_unique_prompts_with_triggers": lambda *a, **kw: [],
        "build_prompt_with_triggers": lambda *a, **kw: ("", ""),
        "clear_trigger_caches": lambda *a, **kw: None,
    },
    "batch_encoding": {
        "batch_encode_prompts": lambda *a, **kw: None,
        "DEFAULT_TE_BATCH_SIZE": 1,
        "encode_prompt_with_combinators": lambda *a, **kw: None,
        "pinned_clip": lambda clip=None, *a, **kw: __import__("contextlib").nullcontext(clip),
        "encode_pair_cached": lambda *a, **kw: (None, None),
    },
    "manifest_utils": {
        "load_existing_manifest": lambda *a, **kw: {},
        "save_manifest": lambda *a, **kw: None,
        "preserve_disk_comparisons": lambda *a, **kw: None,
        "base_dims_of": lambda item: {},
        "update_manifest_comparisons": lambda *a, **kw: {"comparisons": [], "added": 0, "removed": 0, "total": 0},
    },
    "model_loader": {
        "load_checkpoint": lambda *a, **kw: None,
        "load_loras": lambda *a, **kw: None,
        "cleanup_model_references": lambda *a, **kw: None,
        "get_latent_channels": lambda *a, **kw: 4,
        "get_latent_downscale_ratio": lambda *a, **kw: 8,
        "make_empty_latent": lambda *a, **kw: {"samples": None},
        "load_loras_for_preencoding": lambda *a, **kw: None,
        "print_incompatible_loras_summary": lambda *a, **kw: None,
        "load_diffusion_model_and_clip": lambda *a, **kw: (None, None),
        "load_vae_by_name": lambda *a, **kw: None,
    },
    "lora_utils": {
        "expand_lora_folder": lambda *a, **kw: [],
    },
    "image_generation": {
        "generate_image": lambda *a, **kw: None,
        "flush_batch_with_vae": lambda *a, **kw: None,
        "flush_batch_with_remote_vae": lambda *a, **kw: None,
        "create_image_metadata": lambda *a, **kw: {},
        "decode_latent_with_vae": lambda *a, **kw: None,
        "pil_to_image_array": lambda *a, **kw: None,
        "calculate_eta": lambda *a, **kw: 0.0,
        "print_generation_progress": lambda *a, **kw: None,
    },
    "html_generator": {
        "get_html_template": lambda *a, **kw: "",
    },
    "conditioning_cache": {
        "ConditioningCache": type("ConditioningCache", (), {}),
    },
    "remote_vae": {
        "RemoteVAEDecodeWorker": type("RemoteVAEDecodeWorker", (), {}),
        # Facade functions added in B2 — stub returns safe defaults so
        # generation_orchestrator can import and tests run without the
        # companion plugin present.
        "is_remote_vae_available": lambda: False,
        "get_endpoint_names": lambda: ["SD", "SDXL", "Flux", "HunyuanVideo"],
        "INSTALL_INSTRUCTIONS": (
            "Remote VAE requires the ComfyUI-USCG-RemoteVAE companion plugin.\n"
            "Install via Comfy Manager (search 'USCG Remote VAE') or:\n"
            "  git clone https://github.com/JasonHoku/ComfyUI-USCG-RemoteVAE\n"
            "into your ComfyUI/custom_nodes/ directory."
        ),
    },
    "distribution": {
        "INSTALL_INSTRUCTIONS": "stub",
        "is_distribution_available": lambda: False,
        "create_manager": lambda *a, **kw: None,
        "set_active_manager": lambda m: None,
        "clear_active_manager": lambda: None,
        "notify_workers_to_start": lambda *a, **kw: [],
        "stop_all_workers": lambda u: None,
        "get_master_url": lambda: "http://127.0.0.1:8188",
    },
    "civitai": {
        "is_civitai_available": lambda: False,
        "civitai_fetch_by_hash": lambda h: None,
    },
}

import importlib.util as _ilu  # used below for gen_orch, remote_vae, distribution, civitai

for _bare, _attrs in _GEN_ORCH_DEPS.items():
    _fq = f"{_PKG_NAME}.{_bare}"
    if _bare not in sys.modules:
        _s = types.ModuleType(_bare)
        sys.modules[_bare] = _s
    if _fq not in sys.modules:
        sys.modules[_fq] = sys.modules[_bare]
    for _k, _v in _attrs.items():
        setattr(sys.modules[_bare], _k, _v)
        setattr(sys.modules[_fq], _k, _v)
    # Also wire as package attribute
    setattr(sys.modules[_PKG_NAME], _bare, sys.modules[_bare])

# Pre-load generation_orchestrator as part of the package so that relative
# imports (`from .trigger_words import ...`) resolve correctly.  Once loaded
# under the package name it is also registered as a top-level alias so
# `from generation_orchestrator import get_model_cache_key` works.
_GO_PATH = os.path.join(_NODE_ROOT, "generation_orchestrator.py")
_GO_FQ = f"{_PKG_NAME}.generation_orchestrator"
if "generation_orchestrator" not in sys.modules:
    _spec = _ilu.spec_from_file_location(_GO_FQ, _GO_PATH,
                                         submodule_search_locations=[])
    _spec.submodule_search_locations = None  # it is a module, not a package
    _go_mod = _ilu.module_from_spec(_spec)
    _go_mod.__package__ = _PKG_NAME
    sys.modules[_GO_FQ] = _go_mod
    sys.modules["generation_orchestrator"] = _go_mod
    _spec.loader.exec_module(_go_mod)

# Pre-load the real remote_vae module so that test_remote_vae_facade.py gets
# the actual functions (is_remote_vae_available, get_endpoint_names,
# _companion_decode) rather than the lightweight stub registered above.
_RV_PATH = os.path.join(_NODE_ROOT, "remote_vae.py")
_RV_FQ = f"{_PKG_NAME}.remote_vae"
_rv_spec = _ilu.spec_from_file_location(_RV_FQ, _RV_PATH,
                                        submodule_search_locations=[])
_rv_spec.submodule_search_locations = None
_rv_mod = _ilu.module_from_spec(_rv_spec)
_rv_mod.__package__ = _PKG_NAME
# Replace the earlier stub with the real module BEFORE exec so that any
# internal import resolution finds the real object.
sys.modules[_RV_FQ] = _rv_mod
sys.modules["remote_vae"] = _rv_mod
setattr(sys.modules[_PKG_NAME], "remote_vae", _rv_mod)
_rv_spec.loader.exec_module(_rv_mod)

# Pre-load the real distribution.py facade so test_distribution_facade.py exercises
# actual code rather than a stub. (Same pattern as remote_vae loading above —
# Phase 2 addition.)
_DIST_PATH = os.path.join(_NODE_ROOT, "distribution.py")
_DIST_FQ = f"{_PKG_NAME}.distribution"
_dist_spec = _ilu.spec_from_file_location(_DIST_FQ, _DIST_PATH,
                                          submodule_search_locations=[])
_dist_spec.submodule_search_locations = None
_dist_mod = _ilu.module_from_spec(_dist_spec)
_dist_mod.__package__ = _PKG_NAME
sys.modules[_DIST_FQ] = _dist_mod
sys.modules["distribution"] = _dist_mod
setattr(sys.modules[_PKG_NAME], "distribution", _dist_mod)
_dist_spec.loader.exec_module(_dist_mod)

# Pre-load the real civitai.py facade so test_civitai_facade.py exercises
# actual code rather than a stub. (Same pattern as remote_vae + distribution.)
_CIVITAI_PATH = os.path.join(_NODE_ROOT, "civitai.py")
_CIVITAI_FQ = f"{_PKG_NAME}.civitai"
_civitai_spec = _ilu.spec_from_file_location(_CIVITAI_FQ, _CIVITAI_PATH,
                                              submodule_search_locations=[])
_civitai_spec.submodule_search_locations = None
_civitai_mod = _ilu.module_from_spec(_civitai_spec)
_civitai_mod.__package__ = _PKG_NAME
sys.modules[_CIVITAI_FQ] = _civitai_mod
sys.modules["civitai"] = _civitai_mod
setattr(sys.modules[_PKG_NAME], "civitai", _civitai_mod)
_civitai_spec.loader.exec_module(_civitai_mod)


# -- Real-torch probe: opt-in skip for tests marked `requires_torch` ----------
import pytest  # noqa: E402


def _has_real_torch():
    t = sys.modules.get("torch")
    return (t is not None and isinstance(getattr(t, "__file__", None), str)
            and callable(getattr(t, "zeros", None)) and hasattr(t, "inference_mode"))


def pytest_collection_modifyitems(config, items):
    if _has_real_torch():
        return
    if os.environ.get("USCG_REQUIRE_TORCH") == "1":
        return  # let them run and FAIL loudly: the torch job must never skip silently
    skip = pytest.mark.skip(reason="needs real torch (conftest torch stub active); "
                                   "run with ComfyUI's python or the pytest-torch CI job")
    for item in items:
        if item.get_closest_marker("requires_torch"):
            item.add_marker(skip)
