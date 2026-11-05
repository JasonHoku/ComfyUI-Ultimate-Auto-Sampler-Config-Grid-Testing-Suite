"""Roadmap "already implemented, needs testing" items, pinned by tests.

1. Builder VAE settings beat the sampler node's remote_vae_endpoint
2. batch encoding stops on interrupt (before each prompt)
3. pre-encoding only encodes prompts of jobs that still need work
4. omit-trigger matching + the "omits without trigger lookup" warning
5. Builder list order (the result of a drag reorder) survives into configs_json
"""
import contextlib
import importlib.util
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_builder_node import UltimateConfigBuilder
from config_utils import build_job_plan

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load_real(name):
    """The real module under the package name (conftest registers a stub trigger_words for the orchestrator)."""
    lu = sys.modules.get(f"{_PKG}.lora_utils")
    if lu is not None and not hasattr(lu, "load_and_save_tags"):   # conftest stubs lora_utils without it
        lu.load_and_save_tags = lambda *a, **k: []
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}_rm_verify", os.path.join(_ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


tw = None  # set below, after _ROOT/_PKG exist
tw = _load_real("trigger_words")


# --------------------------------------------------------------------------- 1. VAE precedence
def test_normalize_config_vae_treats_blank_and_none_as_default():
    for v in (None, "", "None", "none"):
        assert go.normalize_config_vae(v) == "Default"
    assert go.normalize_config_vae("Default") == "Default"
    assert go.normalize_config_vae("sdxl_vae.safetensors") == "sdxl_vae.safetensors"
    assert go.normalize_config_vae("remote:http://x") == "remote:http://x"


@pytest.fixture
def flush_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(go, "flush_batch_with_vae", lambda *a, **k: calls.append("local"))
    monkeypatch.setattr(go, "flush_batch_with_remote_vae", lambda batch, worker, *a, **k: calls.append(("remote", worker)))
    return calls


def _flush(config_overrides_vae, *, per_config_remote=False, url="http://cfg", workers=None,
           use_remote_vae=True, global_worker="GLOBAL"):
    go._flush_pending_batch(
        [1], per_config_remote, url if per_config_remote else None,
        workers if workers is not None else {}, use_remote_vae, global_worker,
        "LOADED_VAE", {"images": "i"}, {"items": []}, "s", "m", "uid",
        config_overrides_vae=config_overrides_vae)


def test_default_config_vae_uses_sampler_node_remote_endpoint(flush_calls):
    _flush(False)
    assert flush_calls == [("remote", "GLOBAL")]


def test_config_local_vae_beats_sampler_node_remote_endpoint(flush_calls):
    """Builder picks a local VAE file: decode locally even though the node has remote_vae_endpoint set."""
    _flush(True)
    assert flush_calls == ["local"]


def test_config_remote_vae_beats_sampler_node_remote_endpoint(flush_calls):
    _flush(True, per_config_remote=True, workers={"http://cfg": "CFGWORKER"})
    assert flush_calls == [("remote", "CFGWORKER")]


def test_config_remote_vae_without_worker_falls_back_local(flush_calls):
    _flush(True, per_config_remote=True, workers={})
    assert flush_calls == ["local"]


def test_no_remote_endpoint_decodes_locally(flush_calls):
    _flush(False, use_remote_vae=False, global_worker=None)
    assert flush_calls == ["local"]


def test_remote_vae_prefix_helpers():
    assert go.is_remote_vae("remote:http://h/vae")
    assert not go.is_remote_vae("vae.safetensors") and not go.is_remote_vae(None)


# --------------------------------------------------------------------------- 2. interrupt
_spec = importlib.util.spec_from_file_location(f"{_PKG}.batch_encoding_rm_verify", os.path.join(_ROOT, "batch_encoding.py"))
be = importlib.util.module_from_spec(_spec)
be.__package__ = _PKG
_spec.loader.exec_module(be)
if not hasattr(be.torch, "no_grad"):
    be.torch = types.SimpleNamespace(no_grad=contextlib.nullcontext,
                                     cuda=types.SimpleNamespace(is_available=lambda: False))


class _Interrupt(Exception):
    pass


class _Cache:
    def __init__(self):
        self.store = {}

    def get(self, prompt, kind):
        return self.store.get((prompt, kind))

    def set(self, prompt, cond, kind):
        self.store[(prompt, kind)] = cond


def _setup_encode(monkeypatch, interrupt_after):
    """processing_interrupted() turns True once `interrupt_after` prompts have been encoded."""
    encoded = []

    def fake_encode(clip, prompt, clip_skip=0):
        encoded.append(prompt)
        return f"cond:{prompt}"

    fake_mm = types.SimpleNamespace(
        processing_interrupted=lambda: len(encoded) >= interrupt_after,
        InterruptProcessingException=_Interrupt,
    )
    monkeypatch.setattr(be.comfy, "model_management", fake_mm, raising=False)
    monkeypatch.setattr(be, "encode_prompt_with_combinators", fake_encode)
    monkeypatch.setattr(be, "pinned_clip", lambda *a, **k: contextlib.nullcontext())
    return encoded


def test_batch_encode_stops_on_interrupt_before_next_prompt(monkeypatch):
    encoded = _setup_encode(monkeypatch, interrupt_after=1)
    cache = _Cache()
    with pytest.raises(_Interrupt):
        be.batch_encode_with_cache(object(), ["a", "b", "c", "d"], cache, te_batch_size=1)
    assert encoded == ["a"]                    # b, c, d never encoded
    assert list(cache.store) == [("a", "positive")]


def test_batch_encode_interrupt_before_first_prompt_encodes_nothing(monkeypatch):
    encoded = _setup_encode(monkeypatch, interrupt_after=0)
    with pytest.raises(_Interrupt):
        be.batch_encode_with_cache(object(), ["a", "b"], _Cache(), te_batch_size=1)
    assert encoded == []


def test_batch_encode_without_interrupt_encodes_all_and_skips_cached(monkeypatch):
    encoded = _setup_encode(monkeypatch, interrupt_after=10**9)
    cache = _Cache()
    cache.set("b", "cond:b-cached", "positive")
    out = be.batch_encode_with_cache(object(), ["a", "b", "c"], cache, te_batch_size=1)
    assert encoded == ["a", "c"]
    assert out == {"a": "cond:a", "b": "cond:b-cached", "c": "cond:c"}


def test_interrupt_crosses_batch_boundary(monkeypatch):
    """batch_size=2 -> the second batch must still honour the interrupt."""
    encoded = _setup_encode(monkeypatch, interrupt_after=3)
    with pytest.raises(_Interrupt):
        be.batch_encode_with_cache(object(), list("abcdef"), _Cache(), batch_size=2, te_batch_size=1)
    assert encoded == ["a", "b", "c"]


# --------------------------------------------------------------------------- 3. encode only what will run
def _conf(pos, seed=1, **kw):
    c = {"model": "m", "lora": "None", "lora_expanded": "None", "positive": pos, "negative": "neg",
         "seed": seed, "sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 7.0,
         "denoise": 1.0, "vae": "Default"}
    c.update(kw)
    return c


def _item_for(conf, w, h, batch_idx=0):
    return {"seed": conf["seed"], "width": w, "height": h, "batch_idx": batch_idx, "sampler": "euler",
            "scheduler": "normal", "steps": 20, "cfg": 7.0, "denoise": 1.0, "model": "m",
            "positive": conf["positive"], "negative": conf["negative"], "lora": "None"}


def _status_fn(items, *, has_optional_inputs=False):
    def status(pj):
        c = pj["conf"]
        done = go.check_if_job_completed(items, c, c["seed"], pj["w"], pj["h"], pj["batch_idx"],
                                         c["positive"], c["negative"], has_optional_inputs=has_optional_inputs) != -1
        return "exists" if done else "todo"
    return status


def _plan(confs, resolutions=((512, 512),)):
    jobs = [{"label": f"{w}x{h}", "width": w, "height": h, "latent": None, "batch_idx": 0} for w, h in resolutions]
    return build_job_plan(confs, jobs)


def test_only_unfinished_configs_contribute_prompts():
    confs = [_conf("done one"), _conf("todo one"), _conf("done two"), _conf("todo two")]
    items = [_item_for(confs[0], 512, 512), _item_for(confs[2], 512, 512)]
    needed = go.configs_needing_work(_plan(confs), _status_fn(items))
    assert [c["positive"] for c in needed] == ["todo one", "todo two"]
    pos, neg = tw.collect_unique_prompts_with_triggers(needed, "None")
    assert pos == {"todo one", "todo two"}
    assert neg == {"neg"}


def test_all_done_means_nothing_to_encode():
    confs = [_conf("a"), _conf("b")]
    items = [_item_for(c, 512, 512) for c in confs]
    assert go.configs_needing_work(_plan(confs), _status_fn(items)) == []


def test_config_with_one_unfinished_resolution_still_needs_its_prompt():
    confs = [_conf("multi res")]
    items = [_item_for(confs[0], 512, 512)]          # 768x768 missing
    needed = go.configs_needing_work(_plan(confs, ((512, 512), (768, 768))), _status_fn(items))
    assert needed == confs
    assert len(needed) == 1                          # listed once even though two plan entries exist


def test_changed_setting_makes_config_todo_again():
    c = _conf("p")
    items = [_item_for(c, 512, 512)]
    changed = _conf("p", steps=30)
    assert go.configs_needing_work(_plan([changed]), _status_fn(items)) == [changed]


def test_optional_inputs_disable_skipping_so_everything_is_encoded():
    """Documented limitation: with optional inputs connected nothing can be proven unchanged upstream."""
    confs = [_conf("a"), _conf("b")]
    items = [_item_for(c, 512, 512) for c in confs]
    needed = go.configs_needing_work(_plan(confs), _status_fn(items, has_optional_inputs=True))
    assert needed == confs


# --------------------------------------------------------------------------- 4. omit logic
OMIT = tw._should_omit_trigger


def test_omit_is_token_based_not_substring():
    assert OMIT("blue", ["blue"])
    assert OMIT("blue eyes", ["blue"])            # token inside a multi-word trigger
    assert not OMIT("blueberry", ["blue"])
    assert not OMIT("deep blue2", ["blue"])
    assert OMIT("sky-blue", ["blue"])             # hyphen is a boundary


def test_omit_multiword_term_and_case():
    assert OMIT("a red dress", ["red dress"])
    assert not OMIT("red  dress", ["red dress"])  # exact phrase only
    assert OMIT("Blue Eyes".lower(), ["blue eyes"])
    assert OMIT("BLUE", ["blue"])                 # re.IGNORECASE as a second line of defence


def test_omit_term_with_punctuation_matches():
    assert OMIT("(masterpiece)", ["(masterpiece)"])   # \b-based matching failed this one
    assert OMIT("score_9", ["score_9"])
    assert not OMIT("score_95", ["score_9"])


def test_blank_omit_term_never_omits_everything():
    assert not OMIT("anything", [""])
    assert not OMIT("anything", ["", ""])
    assert OMIT("anything", ["", "anything"])


def test_no_omit_list_keeps_everything():
    assert not OMIT("anything", [])


@pytest.fixture
def fake_tags(monkeypatch):
    tags = {"lora_a": ["Blue Eyes,", "  masterpiece ", "blueberry", "(best quality)"]}
    monkeypatch.setattr(tw, "load_and_save_tags", lambda name, force_fetch=False: tags.get(name, []))
    monkeypatch.setattr(tw, "parse_lora_definition", lambda s: [(s.split(":")[0], 1.0, 1.0)] if s != "None" else [])
    tw._get_filtered_lora_triggers_cached.cache_clear()
    tw._build_prompt_cache.clear()
    return tags


def test_filtered_triggers_normalise_whitespace_commas_and_case(fake_tags):
    got = tw.get_filtered_lora_triggers("lora_a:1:1", ["  BLUE EYES, ", "(Best Quality)"])
    assert got == ["masterpiece", "blueberry"]


def test_filtered_triggers_lookup_off_returns_nothing(fake_tags):
    assert tw.get_filtered_lora_triggers("lora_a:1:1", [], lookup_triggers=False) == []
    assert tw.get_filtered_lora_triggers("None", []) == []


def test_build_prompt_applies_omit_in_prompt(fake_tags):
    conf = {"lora": "lora_a:1:1", "positive": "a cat", "lora_omit_triggers": ["blue"]}
    prompt, _ = tw.build_prompt_with_triggers(conf, "Append To End")
    assert "Blue Eyes" not in prompt and "blueberry" in prompt and "masterpiece" in prompt
    pos, _ = tw.collect_unique_prompts_with_triggers([dict(conf, negative="")], "Append To End")
    assert pos == {prompt}


def test_omit_warning_when_lookup_is_off():
    confs = [{"lora": "x:1:1", "lora_omit_triggers": ["blue"]}]
    msg = tw.omit_without_lookup_warning(confs, "None")
    assert msg and "omit" in msg and "1 config" in msg


def test_omit_warning_silent_when_lookup_on_or_nothing_to_omit():
    confs = [{"lora": "x:1:1", "lora_omit_triggers": ["blue"]}]
    for mode in ("Append To End", "Append To Start", "Read From Config"):
        assert tw.omit_without_lookup_warning(confs, mode) is None
    assert tw.omit_without_lookup_warning([{"lora": "x:1:1", "lora_omit_triggers": []}], "None") is None
    assert tw.omit_without_lookup_warning([{"lora": "None", "lora_omit_triggers": ["blue"]}], "None") is None
    assert tw.omit_without_lookup_warning([{"lora": "x:1:1"}], "None") is None


# --------------------------------------------------------------------------- 5. Builder list order (drag reorder result)
def _state(models, loras, vaes):
    return {
        "session_name": "s", "include_none": False, "global_positive_groups": [], "global_negative": "",
        "config_arrays": [{
            "name": "T", "samplers": ["euler"], "schedulers": ["normal"], "steps": "20", "cfg": "7.0",
            "models": models, "vaes": vaes, "text_encoders": [], "clip_type": "stable_diffusion",
            "gguf_options": {}, "loras": loras, "lora_omit_triggers": [],
            "lora_triggerwords_append_settings": {}, "lora_bypass_states": {}, "lora_strength_lock": {},
            "lora_weight_arrays": {}, "model_bypass_states": {}, "vae_bypass_states": {},
            "te_bypass_states": {}, "combine": False, "positive_prompt_groups": [], "negative_prompt": "",
            "use_custom_prompts": False, "model_prompt_prefix": "", "model_prompt_suffix": "",
            "attention_modes": ["default"],
        }],
    }


def _configs(state):
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["configs"]


def test_model_order_in_state_is_order_in_configs_json():
    """One config entry carries the model list; its order is the Builder order (= run order)."""
    def order(models):
        return _configs(_state(models, ["None"], ["None"]))[0]["model"]
    assert order(["m1.safetensors", "m2.safetensors", "m3.safetensors"]) == ["m1.safetensors", "m2.safetensors", "m3.safetensors"]
    assert order(["m3.safetensors", "m1.safetensors", "m2.safetensors"]) == ["m3.safetensors", "m1.safetensors", "m2.safetensors"]


def test_lora_order_in_state_is_lora_stack_and_tag_order_in_configs_json():
    """Stacked LoRAs are joined in Builder order, which is also the trigger-tag append order."""
    def stack(loras):
        return _configs(_state(["m.safetensors"], loras, ["None"]))[0]["lora"]
    assert stack(["a:1:1", "b:1:1", "c:1:1"]) == "a:1:1 + b:1:1 + c:1:1"
    assert stack(["c:1:1", "a:1:1", "b:1:1"]) == "c:1:1 + a:1:1 + b:1:1"
