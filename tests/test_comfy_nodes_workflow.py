"""Roadmap 71: pure-nodes ComfyUI workflow embedded in saved PNG / WEBP files."""
import json
import os
import struct
import sys
import threading

import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import comfy_nodes_workflow as cnw  # noqa: E402

_FIX = os.path.join(_ROOT, "tests", "fixtures")


def _load(name):
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return json.load(f)


ITEMS = _load("nodes_workflow_items.json")
GOLDEN = _load("nodes_workflow_golden.json")


@pytest.mark.parametrize("idx", range(len(ITEMS)))
def test_build_matches_js_golden(idx):
    wf = cnw.build_nodes_workflow(ITEMS[idx])
    wf.pop("id")
    # compare through JSON so int/float spellings (1 vs 1.0) match what JSON.stringify produced
    assert json.loads(json.dumps(wf)) == GOLDEN[idx]


def test_workflow_id_is_stable_uuid():
    a = cnw.build_nodes_workflow(ITEMS[0])["id"]
    assert a == cnw.build_nodes_workflow(ITEMS[0])["id"]
    assert a != cnw.build_nodes_workflow(ITEMS[1])["id"]
    assert cnw.build_nodes_workflow(ITEMS[0], workflow_id="x")["id"] == "x"


def test_js_truthiness_and_parsefloat():
    assert cnw._or(0, 5) == 5 and cnw._or("", 5) == 5 and cnw._or(None, 5) == 5
    assert cnw._or(float("nan"), 5) == 5 and cnw._or("0", 5) == "0" and cnw._or(3, 5) == 3
    assert cnw._parse_float_js("0.8abc") == 0.8
    assert cnw._parse_float_js("abc") is None
    assert cnw._parse_float_js(" 1e2") == 100
    assert cnw._parse_float_js(".5") == 0.5


def test_api_prompt_links_and_widget_names():
    wf = cnw.build_nodes_workflow(ITEMS[2])      # 3 loras
    pr = cnw.workflow_to_api_prompt(wf)
    assert pr["1"]["class_type"] == "CheckpointLoaderSimple"
    assert pr["1"]["inputs"] == {"ckpt_name": "m.safetensors"}
    assert pr["2"]["inputs"]["lora_name"] == "a.safetensors"
    assert pr["2"]["inputs"]["model"] == ["1", 0] and pr["2"]["inputs"]["clip"] == ["1", 1]
    assert pr["3"]["inputs"]["model"] == ["2", 0]
    ks = pr["8"]
    assert ks["class_type"] == "KSampler"
    assert set(ks["inputs"]) == {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
                                 "model", "positive", "negative", "latent_image"}
    assert "control_after_generate" not in ks["inputs"]
    assert ks["inputs"]["model"] == ["4", 0] and ks["inputs"]["positive"] == ["5", 0]
    assert pr["10"]["inputs"] == {"images": ["9", 0]}
    assert pr["9"]["inputs"]["vae"] == ["1", 2]
    assert pr["6"]["inputs"]["clip"] == ["4", 1] and pr["5"]["inputs"]["clip"] == ["4", 1]


def test_note_node_excluded_from_prompt():
    item = dict(ITEMS[0], upscaled=True, upscale_mode="model_only", upscale_model="4x.pth",
                upscale_ratio=1.5, upscale_denoise=0.3, upscale_pipeline="P1")
    wf, pr = cnw.nodes_metadata_for_item(item)
    notes = [n for n in wf["nodes"] if n["type"] == "Note"]
    assert len(notes) == 1 and "model_only" in notes[0]["widgets_values"][0]
    assert "(pipeline P1)" in notes[0]["widgets_values"][0]
    assert all(v["class_type"] != "Note" for v in pr.values())
    assert wf["last_node_id"] == max(n["id"] for n in wf["nodes"])


def test_upscaled_item_uses_base_dims():
    item = dict(ITEMS[0], upscaled=True, width=1248, height=1824, base_width=832, base_height=1216)
    wf, pr = cnw.nodes_metadata_for_item(item)
    lat = [v for v in pr.values() if v["class_type"] == "EmptyLatentImage"][0]
    assert (lat["inputs"]["width"], lat["inputs"]["height"]) == (832, 1216)


@pytest.mark.parametrize("extra", [
    {"media_type": "video"}, {"florence2_no_detection": True}, {"model_type": "workflow"},
    {"model_type": "gguf"}, {"model_type": "diffusion_model"}, {"model_type": "ltx_video"},
])
def test_nodes_metadata_skips(extra):
    assert cnw.nodes_metadata_for_item(dict(ITEMS[0], **extra)) is None


def test_nodes_metadata_checkpoint_ok():
    assert cnw.nodes_metadata_for_item(dict(ITEMS[0], model_type="checkpoint")) is not None
    assert cnw.nodes_metadata_for_item(ITEMS[0]) is not None


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def _png(tmp_path, name="a.png", mode="RGB"):
    p = str(tmp_path / name)
    im = Image.new(mode, (24, 16), (10, 120, 200) if mode == "RGB" else (10, 120, 200, 128))
    im.putpixel((3, 3), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 255))
    im.save(p, format="PNG")
    return p


def test_png_roundtrip_pixels_unchanged_and_replace(tmp_path):
    p = _png(tmp_path)
    before = list(Image.open(p).getdata())
    wf, pr = cnw.nodes_metadata_for_item(ITEMS[1])
    assert cnw.embed_comfy_workflow(p, wf, pr) is True
    assert list(Image.open(p).getdata()) == before
    md = cnw.read_comfy_metadata(p)
    assert md["workflow"] == json.loads(json.dumps(wf)) and md["prompt"] == json.loads(json.dumps(pr))
    # PIL sees the text chunks too (what ComfyUI's own loader uses)
    assert "workflow" in Image.open(p).text
    wf2, pr2 = cnw.nodes_metadata_for_item(ITEMS[3])
    cnw.embed_comfy_workflow(p, wf2, pr2)
    raw = open(p, "rb").read()
    assert raw.count(b"tEXt") == 2                      # replaced, not duplicated
    assert cnw.read_comfy_metadata(p)["workflow"]["nodes"][0]["widgets_values"] == ["m.safetensors"]
    assert list(Image.open(p).getdata()) == before


def test_png_rgba_ok(tmp_path):
    p = _png(tmp_path, "b.png", "RGBA")
    before = list(Image.open(p).getdata())
    wf, pr = cnw.nodes_metadata_for_item(ITEMS[0])
    assert cnw.embed_comfy_workflow(p, wf, pr)
    assert list(Image.open(p).getdata()) == before


# --------------------------------------------------------------------------- #
# WEBP
# --------------------------------------------------------------------------- #
def _webp(tmp_path, name, mode="RGB", **kw):
    p = str(tmp_path / name)
    im = Image.new(mode, (40, 30), (30, 90, 160) if mode == "RGB" else (30, 90, 160, 100))
    im.save(p, format="WEBP", **kw)
    return p


def _riff_names(p):
    return [c[0] for c in cnw._riff_chunks(open(p, "rb").read())]


def _check_webp(p, item):
    size = Image.open(p).size
    wf, pr = cnw.nodes_metadata_for_item(item)
    assert cnw.embed_comfy_workflow(p, wf, pr) is True
    data = open(p, "rb").read()
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8       # RIFF size patched
    assert len(data) % 2 == 0
    md = cnw.read_comfy_metadata(p)
    assert md["workflow"] == json.loads(json.dumps(wf)) and md["prompt"] == json.loads(json.dumps(pr))
    ex = Image.open(p).getexif()
    assert ex[0x010F].startswith("workflow:") and ex[0x0110].startswith("prompt:")
    im = Image.open(p)
    im.load()                                                        # still decodes
    assert im.size == size
    names = _riff_names(p)
    assert names[0] == b"VP8X" and names.count(b"EXIF") == 1
    flags = cnw._riff_chunks(data)[0][1][0]
    assert flags & 0x08
    return names, flags


def test_webp_lossy_rgb_vp8(tmp_path):
    p = _webp(tmp_path, "a.webp", quality=80)
    assert _riff_names(p)[0] == b"VP8 "
    _check_webp(p, ITEMS[2])


def test_webp_rgba_lossless_vp8l(tmp_path):
    p = _webp(tmp_path, "b.webp", "RGBA", lossless=True)
    assert _riff_names(p)[0] == b"VP8L"
    names, flags = _check_webp(p, ITEMS[1])
    assert flags & 0x10


def test_webp_rgba_lossy_alph(tmp_path):
    p = _webp(tmp_path, "c.webp", "RGBA", quality=70)
    names, flags = _check_webp(p, ITEMS[1])
    assert b"ALPH" in names and flags & 0x10


def test_webp_existing_vp8x_and_reembed(tmp_path):
    p = _webp(tmp_path, "d.webp", quality=80)
    cnw.embed_webp_exif(p, {cnw.EXIF_MAKE: "workflow:{}"})          # now a VP8X container
    assert _riff_names(p)[0] == b"VP8X"
    _check_webp(p, ITEMS[0])
    _check_webp(p, ITEMS[3])                                        # replaces the EXIF chunk
    assert _riff_names(p).count(b"EXIF") == 1


def test_non_ascii_prompt_survives(tmp_path):
    item = dict(ITEMS[0], positive="猫 ✨ cat")
    for kind in ("png", "webp"):
        p = _png(tmp_path, "u.png") if kind == "png" else _webp(tmp_path, "u.webp", quality=80)
        wf, pr = cnw.nodes_metadata_for_item(item)
        assert cnw.embed_comfy_workflow(p, wf, pr)
        md = cnw.read_comfy_metadata(p)
        texts = [v["inputs"].get("text") for v in md["prompt"].values() if v["class_type"] == "CLIPTextEncode"]
        assert "猫 ✨ cat" in texts


def test_jpeg_returns_false_and_untouched(tmp_path):
    p = str(tmp_path / "a.jpg")
    Image.new("RGB", (16, 16), (1, 2, 3)).save(p, format="JPEG")
    before = open(p, "rb").read()
    wf, pr = cnw.nodes_metadata_for_item(ITEMS[0])
    assert cnw.embed_comfy_workflow(p, wf, pr) is False
    assert open(p, "rb").read() == before
    assert cnw.stamp_file(p, ITEMS[0], enabled=True) is False


def test_stamp_file_never_raises(tmp_path):
    p = str(tmp_path / "bad.png")
    with open(p, "wb") as f:
        f.write(cnw.PNG_SIG + b"garbage")
    assert cnw.stamp_file(p, ITEMS[0], enabled=True) is False
    assert cnw.stamp_file(str(tmp_path / "missing.webp"), ITEMS[0], enabled=True) is False
    assert cnw.stamp_file(p, None, enabled=True) is False


def test_stamp_file_enabled_flag_and_context(tmp_path):
    p = _png(tmp_path, "s.png")
    assert cnw.enabled() is False
    assert cnw.stamp_file(p, ITEMS[0]) is False                    # no run: off
    assert cnw.stamp_file(p, ITEMS[0], enabled=False) is False
    assert cnw.read_comfy_metadata(p)["workflow"] is None
    tok = cnw.begin_run(True)
    try:
        assert cnw.enabled() is True
        assert cnw.stamp_file(p, ITEMS[0]) is True
    finally:
        cnw.end_run(tok)
    assert cnw.read_comfy_metadata(p)["workflow"] is not None
    assert cnw.enabled() is False


def test_run_context_nested_and_threads():
    a = cnw.begin_run(True)
    b = cnw.begin_run(False)
    assert cnw.enabled() is False
    cnw.end_run(a)                       # closing the older run must not clobber the newer one
    assert cnw.enabled() is False
    cnw.end_run(b)
    assert cnw.enabled() is False
    cnw.end_run(b)                       # idempotent
    seen = {}
    t1 = cnw.begin_run(True)

    def other():
        seen["during"] = cnw.enabled()   # another thread sees the process-wide open run
        t2 = cnw.begin_run(False)
        seen["nested"] = cnw.enabled()
        cnw.end_run(t2)

    th = threading.Thread(target=other)
    th.start()
    th.join()
    assert seen == {"during": True, "nested": False}
    assert cnw.enabled() is True
    cnw.end_run(t1)
    assert cnw.enabled() is False


def test_source_has_no_forbidden_calls():
    with open(os.path.join(_ROOT, "comfy_nodes_workflow.py"), encoding="utf-8") as f:
        src = f.read()
    for bad in ("urllib", "requests", "subprocess", "eval(", "exec(", "import piexif"):
        assert bad not in src
