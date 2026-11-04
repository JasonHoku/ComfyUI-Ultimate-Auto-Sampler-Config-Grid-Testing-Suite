"""Pure backend of GET /config_tester/manifest (run_api.load_manifest_for_fetch)."""
import importlib.util
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}_real_mf", os.path.join(_ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


run_api = _load("run_api", "run_api.py")
fetch = run_api.load_manifest_for_fetch


@pytest.fixture
def bench(tmp_path):
    base = tmp_path / "benchmarks"
    (base / "sess1").mkdir(parents=True)
    manifest = {"session_name": "sess1", "meta": {"model": "m"},
                "items": [{"id": 2, "note": "café ❤ </script>"}, {"id": 1, "note": ""}]}
    with open(base / "sess1" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "manifest.json").write_text('{"items": []}')
    return str(base), manifest


@pytest.mark.parametrize("raw", ["", "   ", None, "..", "../", "//", "<>"])
def test_empty_after_sanitising_is_400(bench, raw):
    assert fetch(bench[0], raw)[0] == 400


@pytest.mark.parametrize("raw", ["../secret", "..\\secret", "a/../../secret", "<script>", "sess1/../../secret"])
def test_never_reads_outside_benchmarks(bench, raw):
    status, body, _ = fetch(bench[0], raw)
    assert status in (400, 403, 404)
    assert b"items" not in (body or b"")


def test_junction_or_symlink_escape_is_403(bench, tmp_path):
    base, _ = bench
    link = os.path.join(base, "evil")
    try:
        os.symlink(str(tmp_path / "secret"), link, target_is_directory=True)
    except (OSError, NotImplementedError):
        import subprocess
        if os.name != "nt" or subprocess.run(["cmd", "/c", "mklink", "/J", link, str(tmp_path / "secret")],
                                             capture_output=True).returncode != 0:
            pytest.skip("cannot create a symlink or junction here")
    assert fetch(base, "evil")[0] == 403


def test_missing_is_404_json(bench):
    status, body, _ = fetch(bench[0], "nope")
    assert status == 404 and json.loads(body) == {"error": "not found"}


def test_ok_body_is_compact_and_equal(bench):
    base, manifest = bench
    status, body, headers = fetch(base, "sess1")
    assert status == 200
    assert json.loads(body.decode("utf-8")) == manifest
    assert b"\n" not in body and b": " not in body.replace("café".encode(), b"")
    assert "café ❤".encode("utf-8") in body  # ensure_ascii=False keeps unicode
    assert headers["Cache-Control"] == "no-cache" and headers["ETag"].startswith('W/"')
    assert len(body) < os.path.getsize(os.path.join(base, "sess1", "manifest.json"))


def test_etag_stable_then_changes_on_rewrite(bench):
    base, manifest = bench
    e1 = fetch(base, "sess1")[2]["ETag"]
    assert fetch(base, "sess1")[2]["ETag"] == e1
    path = os.path.join(base, "sess1", "manifest.json")
    manifest["items"].append({"id": 3})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert fetch(base, "sess1")[2]["ETag"] != e1


def test_if_none_match_variants(bench):
    base, _ = bench
    etag = fetch(base, "sess1")[2]["ETag"]
    for inm in (etag, etag[2:], f'"zzz", {etag}', "*"):
        status, body, headers = fetch(base, "sess1", inm)
        assert status == 304 and body is None and headers["ETag"] == etag, inm
    status, body, _ = fetch(base, "sess1", 'W/"stale"')
    assert status == 200 and body


def test_truncated_file_is_503_with_retry_after(bench):
    base, _ = bench
    path = os.path.join(base, "sess1", "manifest.json")
    raw = open(path, "rb").read()
    open(path, "wb").write(raw[: len(raw) // 2])
    status, _, headers = fetch(base, "sess1")
    assert status == 503 and headers["Retry-After"] == "1"


def test_empty_file_is_503(bench):
    base, _ = bench
    open(os.path.join(base, "sess1", "manifest.json"), "wb").close()
    assert fetch(base, "sess1")[0] == 503
