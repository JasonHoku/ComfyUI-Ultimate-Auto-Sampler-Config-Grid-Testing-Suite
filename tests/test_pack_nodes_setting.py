"""Embed ComfyUI Nodes Workflow run setting: Builder emit -> manifest meta -> run context -> every save site,
plus the master's pass over worker-rendered images."""
import json
import os
import re
import sys

import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import comfy_nodes_workflow as cnw  # noqa: E402
from config_builder_node import UltimateConfigBuilder  # noqa: E402
from config_utils import run_meta_extras  # noqa: E402
from tests.test_state_to_configs_json import make_state  # noqa: E402
from tests.test_random_seeds_setting import _load_sampler_node, _run_tests_arg  # noqa: E402


def _settings(**kw):
    state = make_state()
    state.update(kw)
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["_session_settings"]


# ---- Builder emit ------------------------------------------------------------------------
def test_emit_true_false_and_default():
    assert _settings(pack_nodes_metadata=True)["pack_nodes_metadata"] is True
    assert _settings(pack_nodes_metadata=False)["pack_nodes_metadata"] is False
    assert _settings()["pack_nodes_metadata"] is False          # always emitted, default off


def test_emit_garbage_is_falsy():
    assert _settings(pack_nodes_metadata=None)["pack_nodes_metadata"] is False
    assert _settings(pack_nodes_metadata="")["pack_nodes_metadata"] is False


def test_run_meta_extras_carries_it():
    ss = _settings(pack_nodes_metadata=True)
    assert run_meta_extras(ss, False)["session_settings"]["pack_nodes_metadata"] is True


# ---- sampler_node opens / closes the run context ------------------------------------------
def test_sampler_node_begins_run_with_the_setting(monkeypatch):
    sn = _load_sampler_node()
    calls = []
    monkeypatch.setattr(sn.comfy_nodes_workflow, "begin_run", lambda on: calls.append(("begin", on)) or "tok")
    monkeypatch.setattr(sn.comfy_nodes_workflow, "end_run", lambda tok=None: calls.append(("end", tok)))
    _run_tests_arg(monkeypatch, json.dumps({"configs": [{}], "_session_settings": {"pack_nodes_metadata": True}}))
    assert calls == [("begin", True), ("end", "tok")]
    calls.clear()
    _run_tests_arg(monkeypatch, json.dumps({"configs": [{}], "_session_settings": {}}))
    assert calls == [("begin", False), ("end", "tok")]
    calls.clear()
    _run_tests_arg(monkeypatch, json.dumps({"configs": [{}]}))
    assert calls == [("begin", False), ("end", "tok")]


# ---- every image write has a stamp next to it -----------------------------------------------
_SAVE = re.compile(r"\.save\(")


def test_every_image_save_is_followed_by_a_stamp():
    total = 0
    for name in ("image_generation.py", "remote_vae.py", "generation_orchestrator.py", "upscale_runner.py"):
        with open(os.path.join(_ROOT, name), encoding="utf-8") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if not _SAVE.search(line) or line.lstrip().startswith("#"):
                continue
            if any("def save_image_to_disk" in l for l in lines[max(0, i - 14):i]):
                continue                        # unused generic helper, not a generation save site
            total += 1
            window = "\n".join(lines[i:i + 26])
            assert "stamp_file(" in window, f"{name}:{i + 1} writes an image without comfy_nodes_workflow.stamp_file()"
    assert total >= 12      # 3 + 3 (gen save, remote save) ... sanity: the regex still finds the sites


def test_upscale_runner_stamps_with_the_explicit_session_flag():
    with open(os.path.join(_ROOT, "upscale_runner.py"), encoding="utf-8") as f:
        src = f.read()
    assert "stamp_file(path, meta, enabled=pack)" in src
    assert '"pack_nodes_metadata"' in src            # read from manifest meta, never from run state


# ---- distributed pass --------------------------------------------------------------------------
def test_distributed_pass_stamps_only_new_non_upscaled_items(tmp_path):
    import generation_orchestrator as go
    imgs = tmp_path / "images"
    imgs.mkdir()

    def mk(name, fmt):
        p = imgs / name
        Image.new("RGB", (16, 16), (5, 6, 7)).save(p, format=fmt)
        return p

    p_new = mk("img_1.webp", "WEBP")
    p_png = mk("img_2.png", "PNG")
    p_old = mk("img_3.webp", "WEBP")
    p_up = mk("img_4_upscaled.webp", "WEBP")
    p_jpg = mk("img_5.jpg", "JPEG")
    base = {"model": "m.safetensors", "lora": "None", "positive": "p", "negative": "n", "width": 16, "height": 16,
            "seed": 1, "steps": 5, "cfg": 5, "sampler": "euler", "scheduler": "normal", "denoise": 1}
    items = [
        dict(base, id=1, filename="img_1.webp"),
        dict(base, id=2, filename="img_2.png"),
        dict(base, id=3, filename="img_3.webp"),                       # existed before the run
        dict(base, id=4, filename="img_4_upscaled.webp", upscaled=True),
        dict(base, id=5, filename="img_5.jpg"),                        # JPEG cannot carry it
        dict(base, id=6, media_type="video", filename="v.mp4"),
        dict(base, id=7, filename="img_7.webp", model_type="gguf"),    # not expressible as the plain graph
    ]
    mk("img_7.webp", "WEBP")

    def path_for(item, images_dir):
        name = item.get("filename")
        return os.path.join(images_dir, name) if name and name.lower().endswith((".webp", ".png", ".jpg")) else None

    n = go._distributed_pack_nodes_pass({"items": items}, {3}, str(imgs), image_path_for=path_for)
    assert n == 2
    assert cnw.read_comfy_metadata(str(p_new))["workflow"] is not None
    assert cnw.read_comfy_metadata(str(p_png))["workflow"] is not None
    for untouched in (p_old, p_up, p_jpg, imgs / "img_7.webp"):
        assert cnw.read_comfy_metadata(str(untouched))["workflow"] is None
    assert Image.open(p_new).size == (16, 16)


def test_orchestrator_runs_the_pass_only_when_enabled():
    with open(os.path.join(_ROOT, "generation_orchestrator.py"), encoding="utf-8") as f:
        src = f.read()
    assert "if comfy_nodes_workflow.enabled():\n        _nres = _distributed_pack_nodes_pass(" in src.replace("\r\n", "\n")
    assert '_extras["pack_nodes_metadata"] = True' in src


def test_export_falls_back_to_the_python_builder():
    with open(os.path.join(_ROOT, "__init__.py"), encoding="utf-8") as f:
        src = f.read()
    assert "from .comfy_nodes_workflow import build_nodes_workflow" in src
