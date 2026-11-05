"""Distributed progress events: payload maths, emit helper, on_status hook, format_eta_str."""
import ast
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generation_orchestrator as go  # preloaded by tests/conftest.py


def _real_format_eta_str():
    """image_generation is stubbed under test (needs torch): rebuild the real function from source."""
    with open(os.path.join(ROOT, "image_generation.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "format_eta_str")
    code = compile(ast.Module(body=[node], type_ignores=[]), "format_eta_str", "exec")
    fn_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "format_eta_str")
    return types.FunctionType(fn_code, {})


@pytest.fixture
def fmt(monkeypatch):
    f = _real_format_eta_str()
    monkeypatch.setattr(go, "format_eta_str", f)
    return f


def test_format_eta_str_cases(fmt):
    assert fmt(9) == "9s"
    assert fmt(0) == "0s"
    assert fmt(125) == "2m 5s"
    assert fmt(60) == "1m 0s"
    assert fmt(3725) == "1h 2m"
    assert fmt(7200) == "2h 0m"
    assert fmt(-4) == "0s"
    assert fmt(9.9) == "9s"


BASE = {"pre_skipped": 2, "ltx_done": 1, "plan_total": 10}
STATUS = {"completed": 3, "skipped": 1, "failed": 0, "pending": 3, "claimed": 0, "total": 7}


def test_payload_maths(fmt):
    eta = {"eta_seconds": 125, "avg_duration": 10.26, "active_workers": 2}
    p = go._distributed_progress_payload(STATUS, eta, BASE, "n1", "sess")
    assert p["current_job"] == 2 + 1 + 3 + 1 + 0 == 7
    assert p["total_jobs"] == 10
    assert p["progress_pct"] == 70
    assert p["phase"] == "distributed"
    assert p["workers"] == 2
    assert p["eta_str"] == "2m 5s"
    assert p["avg_duration"] == 10.3
    assert p["node"] == "n1" and p["session_name"] == "sess"
    assert p["last_duration"] == 0
    assert len(p["finish_time"]) == 8


def test_payload_counts_failed_and_caps_at_100(fmt):
    p = go._distributed_progress_payload({"completed": 20, "failed": 2}, None, BASE, "n", "s")
    assert p["progress_pct"] == 100


def test_payload_without_eta(fmt):
    p = go._distributed_progress_payload(STATUS, None, BASE, "n", "s")
    assert p["eta_str"] == "—"
    assert p["finish_time"] == ""
    assert p["workers"] == 0
    p2 = go._distributed_progress_payload(
        STATUS, {"eta_seconds": None, "avg_duration": None, "active_workers": 0}, BASE, "n", "s")
    assert p2["eta_str"] == "—"


class _Server:
    def __init__(self):
        self.events = []
        self.instance = self

    def send_sync(self, name, payload):
        self.events.append((name, payload))


class _Mgr:
    unique_id = "n1"
    session_name = "sess"
    is_active = True

    def __init__(self, with_eta=True):
        self.on_status = None
        if with_eta:
            self.get_eta = lambda: {"eta_seconds": 30, "avg_duration": 5.0, "active_workers": 3}

    def get_status(self):
        return dict(STATUS)


def test_emit_sends_progress_event(monkeypatch, fmt):
    srv = _Server()
    monkeypatch.setattr(go, "PromptServer", srv)
    go._emit_distributed_progress(_Mgr(), BASE, "n1", "sess")
    assert len(srv.events) == 1
    name, p = srv.events[0]
    assert name == "ultimate_grid.progress"
    assert p["workers"] == 3 and p["current_job"] == 7 and p["eta_str"] == "30s"


def test_emit_without_get_eta_and_never_raises(monkeypatch, fmt):
    srv = _Server()
    monkeypatch.setattr(go, "PromptServer", srv)
    go._emit_distributed_progress(_Mgr(with_eta=False), BASE, "n1", "sess")
    assert srv.events[0][1]["eta_str"] == "—"
    monkeypatch.setattr(go, "PromptServer", None)
    go._emit_distributed_progress(_Mgr(), BASE, "n1", "sess")  # no server: no error
    go._emit_distributed_progress(object(), BASE, "n1", "sess")  # broken manager: no error


def test_send_distribution_status_calls_on_status():
    mgr = _Mgr()
    seen = []
    mgr.on_status = seen.append
    go._send_distribution_status(mgr)  # no `server` module under test: the callback still fires
    assert len(seen) == 1 and seen[0]["completed"] == 3 and seen[0]["active"] is True


def test_send_distribution_status_survives_failing_callback():
    mgr = _Mgr()

    def boom(st):
        raise RuntimeError("x")

    mgr.on_status = boom
    go._send_distribution_status(mgr)
