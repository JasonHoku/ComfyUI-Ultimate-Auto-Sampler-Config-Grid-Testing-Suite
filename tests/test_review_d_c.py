"""Review-d (C): the lazy-shell flag must never persist nor be served."""
import importlib.util
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load_run_api():
    spec = importlib.util.spec_from_file_location(f"{_PKG}.run_api_real_rdc", os.path.join(_ROOT, "run_api.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


def test_fetch_heals_flagged_manifest(tmp_path):
    base = tmp_path / "benchmarks"
    (base / "s1").mkdir(parents=True)
    with open(base / "s1" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"session_name": "s1", "meta": {}, "items": [{"id": 1}], "__uscg_lazy__": True}, f)
    status, body, _ = _load_run_api().load_manifest_for_fetch(str(base), "s1")
    assert status == 200
    data = json.loads(body)
    assert "__uscg_lazy__" not in data and data["items"] == [{"id": 1}]


def test_save_manifest_route_drops_flag():
    src = open(os.path.join(_ROOT, "__init__.py"), encoding="utf-8").read()
    seg = src.split("async def save_manifest(request)")[1].split("@server.PromptServer")[0]
    assert 'manifest_data.pop("__uscg_lazy__", None)' in seg
