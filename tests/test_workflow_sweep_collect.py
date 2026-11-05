"""Workflow Sweep: manifest items (build_collect_item) and the disk-fresh manifest_utils.append_items."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.basename(ROOT)
sys.path.insert(0, ROOT)

import workflow_sweep as ws  # noqa: E402


def _load_real(name, private):
    """tests/conftest.py registers STUBS for several modules; load the real file under a private name."""
    spec = importlib.util.spec_from_file_location(f"{PKG}.{private}", os.path.join(ROOT, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = PKG
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest_utils():
    return _load_real("manifest_utils", "manifest_utils_real")


@pytest.fixture(scope="module")
def create_image_metadata():
    torch = sys.modules.get("torch")
    patched = False
    if torch is not None and not hasattr(torch, "inference_mode"):
        torch.inference_mode = lambda *a, **k: (lambda f: f)  # stub torch lacks it; real torch needs no patch
        patched = True
    try:
        return _load_real("image_generation", "image_generation_real").create_image_metadata
    finally:
        if patched:
            del torch.inference_mode


CONF = {"model_type": "workflow", "media_type": "image", "wf_template_hash": "h", "wf_source": "t.json",
        "wf_output": "38:0", "wf_subs": [["32", "cfg", 4.0]], "wf_params": {"KSampler.cfg": 4.0},
        "wf_fingerprint": "f" * 20, "model": "m.safetensors", "sampler": "er_sde", "scheduler": "simple",
        "steps": 25, "cfg": 4.0, "seed": 7, "denoise": 1.0, "positive": "a teapot", "negative": "ugly",
        "lora": "None", "lora_expanded": "None", "vae": "v.safetensors"}


def test_build_collect_item_fields(create_image_metadata):
    item = ws.build_collect_item(CONF, 832, 1216, 12.5, 0, 3, create_image_metadata=create_image_metadata)
    for key, val in {"model": "m.safetensors", "sampler": "er_sde", "scheduler": "simple", "steps": 25, "cfg": 4.0,
                     "seed": 7, "positive": "a teapot", "negative": "ugly", "config_positive": "a teapot",
                     "config_negative": "ugly", "width": 832, "height": 1216, "duration": 12.5, "batch_idx": 0,
                     "gen_index": 3, "model_type": "workflow", "wf_fingerprint": "f" * 20,
                     "favorited": False, "rejected": False, "note": ""}.items():
        assert item[key] == val, key
    assert item["denoise"] == 1 and isinstance(item["denoise"], int)  # 1.0 -> 1, as flush_batch_with_vae
    assert item["wf_params"] == {"KSampler.cfg": 4.0} and item["wf_subs"] == [["32", "cfg", 4.0]]


def test_build_collect_item_manual_use_without_conf(create_image_metadata):
    item = ws.build_collect_item({}, 64, 64, 0, 0, 0, create_image_metadata=create_image_metadata)
    assert item["model_type"] == "workflow" and item["seed"] == 0 and item["positive"] == ""
    assert item["width"] == 64 and item["favorited"] is False


def test_build_collect_item_does_not_mutate_conf(create_image_metadata):
    conf = dict(CONF)
    ws.build_collect_item(conf, 8, 8, 1, 0, 0, create_image_metadata=create_image_metadata)
    assert conf == CONF


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_append_items_inserts_newest_first_and_creates_file(manifest_utils, tmp_path):
    mp = str(tmp_path / "s" / "manifest.json")
    manifest_utils.append_items(mp, [{"id": 1}], session_name="s", default_meta={"model": "x"})
    data = manifest_utils.append_items(mp, [{"id": 2}, {"id": 3}], session_name="s", default_meta={"model": "y"})
    assert [i["id"] for i in data["items"]] == [3, 2, 1]  # each insert(0), like every other writer
    assert data["meta"] == {"model": "x"} and data["session_name"] == "s"
    with open(mp) as f:
        assert json.load(f) == data


def test_append_items_preserves_disk_favorites_and_other_writers(manifest_utils, tmp_path):
    mp = str(tmp_path / "s" / "manifest.json")
    _write(mp, {"session_name": "s", "meta": {"m": 1}, "items": [{"id": 10, "favorited": True, "note": "keep"}]})
    # a standard run (or the user) edits the file between two sweep jobs
    d = manifest_utils.append_items(mp, [{"id": 11}])
    assert d["items"][1] == {"id": 10, "favorited": True, "note": "keep"}
    with open(mp) as f:
        on_disk = json.load(f)
    on_disk["items"][0]["favorited"] = True  # user favourites the new item
    on_disk["items"].append({"id": 5})       # another writer appended something
    _write(mp, on_disk)
    d = manifest_utils.append_items(mp, [{"id": 12}])
    ids = [i["id"] for i in d["items"]]
    assert ids == [12, 11, 10, 5]
    assert next(i for i in d["items"] if i["id"] == 11)["favorited"] is True


def test_append_items_remove_ids_deletes_item_and_file(manifest_utils, tmp_path):
    base = tmp_path / "s"
    mp = str(base / "manifest.json")
    img = base / "images" / "img_old.webp"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    outside = tmp_path / "keep.webp"
    outside.write_bytes(b"x")
    _write(mp, {"items": [{"id": 1, "file": "/view?filename=img_old.webp&type=output&subfolder=benchmarks/s/images"},
                          {"id": 2, "file": "/view?filename=../../keep.webp&type=output&subfolder=x"},
                          {"id": 3}]})
    d = manifest_utils.append_items(mp, [{"id": 9}], remove_ids=[1, 2])
    assert [i["id"] for i in d["items"]] == [9, 3]
    assert not img.exists()
    assert outside.exists()  # never deletes outside <session>/images


def test_append_items_refuses_corrupt_manifest(manifest_utils, tmp_path):
    mp = tmp_path / "s" / "manifest.json"
    mp.parent.mkdir()
    mp.write_text("{not json")
    with pytest.raises(RuntimeError):   # never replace an unreadable manifest with a fresh one (wipes the session)
        manifest_utils.append_items(str(mp), [{"id": 1}], session_name="s")
    assert mp.read_text() == "{not json"


def test_session_paths_sanitises_name(monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    p = ws.session_paths("../../evil name!")
    assert p["session"] == "evilname"
    assert p["base"] == os.path.join(str(tmp_path), "benchmarks", "evilname")
    assert os.path.commonpath([p["manifest"], str(tmp_path / "benchmarks")]) == str(tmp_path / "benchmarks")
    assert ws.session_paths("")["session"] == "default_session"
