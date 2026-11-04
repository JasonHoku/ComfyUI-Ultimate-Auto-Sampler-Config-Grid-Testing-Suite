"""Generator node in workflow mode: IS_CHANGED, hand-off to the sweep runner, never blocks."""
import importlib
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.basename(ROOT)
sys.path.insert(0, ROOT)

# the Generator imports the runner as <package>.workflow_sweep_runner; use that same module instance here
runner = importlib.import_module(f"{PKG}.workflow_sweep_runner")


@pytest.fixture(scope="module")
def sampler_node():
    lora_utils = sys.modules[f"{PKG}.lora_utils"]
    if not hasattr(lora_utils, "load_and_save_tags"):
        lora_utils.load_and_save_tags = lambda *a, **k: None
    sys.modules["comfy"].utils = sys.modules["comfy.utils"]
    sys.modules["comfy"].sd = sys.modules["comfy.sd"]
    # root conftest registers a stub for sampler_node; load the real file under a private name
    spec = importlib.util.spec_from_file_location(f"{PKG}.sampler_node_real", os.path.join(ROOT, "sampler_node.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = PKG
    spec.loader.exec_module(mod)
    return mod


def wf_config():
    with open(os.path.join(ROOT, "tests", "fixtures", "image_moss_image_t2i.json"), encoding="utf-8") as f:
        wf = json.load(f)
    return {"model_type": "workflow", "name": "c", "workflow": wf,
            "workflow_output": {"node_id": "38", "output_index": 0},
            "workflow_bindings": [{"id": "b", "kind": "literal", "targets": [{"node_id": "32", "input": "seed"}],
                                   "values": [1, 2]}]}


def test_workflow_configs_detection(sampler_node):
    detect = sampler_node._workflow_configs
    assert detect(json.dumps({"configs": [wf_config()]})) and detect(json.dumps([wf_config()]))
    assert detect(json.dumps({"configs": [{"model_type": "checkpoint", "model": "workflow.safetensors"}]})) is None
    assert detect("not json workflow") is None and detect(None) is None and detect("{}") is None


def test_is_changed_is_nan_for_workflow_and_stable_for_standard(sampler_node):
    cls = sampler_node.SamplerGridTester
    wf = json.dumps({"configs": [wf_config()]})
    assert cls.IS_CHANGED(configs_json=wf) != cls.IS_CHANGED(configs_json=wf)   # NaN
    std = json.dumps({"configs": [{"model_type": "checkpoint", "model": "a"}]})
    assert cls.IS_CHANGED(configs_json=std) == cls.IS_CHANGED(configs_json=std)


class _FakeBridge:
    def __init__(self):
        self.enqueued = []

    def run_on_loop(self, coro):
        import asyncio
        return asyncio.run(coro)

    async def validate(self, prompt, partial):
        return True, None, {}

    async def enqueue(self, prompt, partial, client_id):
        self.enqueued.append(prompt)
        return f"p{len(self.enqueued)}", None, {}

    def queue_state(self):
        return set(), {f"p{i + 1}" for i in range(len(self.enqueued))}

    def history_entry(self, pid):
        return None

    def delete_queued(self, pid):
        return True

    def interrupt(self, pid):
        return False

    def send_sync(self, event, data):
        pass


def test_run_tests_starts_sweep_and_returns_dashboard_without_waiting(sampler_node, tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda t: ["a.safetensors"], raising=False)
    runner.reset_registry()
    bridge = _FakeBridge()
    runner.set_bridge(bridge)
    captured = {}
    monkeypatch.setattr(sampler_node, "get_html_template",
                        lambda title, manifest, node_id: captured.update(title=title, manifest=manifest, node=node_id) or "<html>")
    monkeypatch.setattr(runner, "TICK", 0.05)
    # class_info=None default arg of start_sweep would hit ComfyUI's registry; the stub `nodes` has none -> None
    cfg = json.dumps({"configs": [wf_config()], "_session_settings": {"session_name": "gen_wf", "start_at_job": 0}})
    try:
        out = sampler_node.SamplerGridTester().run_tests(cfg, unique_id="77")
        assert out == ("<html>",) and captured["title"] == "gen_wf" and captured["node"] == "77"
        assert captured["manifest"]["session_name"] == "gen_wf" and captured["manifest"]["items"] == []
        sw = runner._active_sweep()
        assert sw is not None and sw.origin_node == "77" and len(sw.jobs) == 2   # returned while jobs are pending
        # a second Generator run while the sweep is active is refused with a readable error
        with pytest.raises(RuntimeError, match="already active"):
            sampler_node.SamplerGridTester().run_tests(cfg, unique_id="77")
    finally:
        sw = runner._active_sweep()
        if sw is not None:
            runner.cancel(sw.id)
            sw.thread.join(5)
        runner.set_bridge(None)
        runner.reset_registry()


def test_mixed_workflow_and_standard_configs_raise(sampler_node, tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda t: ["a.safetensors"], raising=False)
    runner.reset_registry()
    runner.set_bridge(_FakeBridge())
    cfg = json.dumps({"configs": [wf_config(), {"model_type": "checkpoint", "model": "x"}]})
    try:
        with pytest.raises(ValueError, match="cannot be mixed"):
            sampler_node.SamplerGridTester().run_tests(cfg, unique_id="1")
    finally:
        runner.set_bridge(None)
        runner.reset_registry()
