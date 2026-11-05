"""Master waits for remote workers inside the claim loop; reclaim + timeout helpers."""
import os
import re
import sys
import threading
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _JobState:
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"


class _Job:
    def __init__(self, job_id, state, worker_id=None, claimed_at=None):
        self.job_id, self.state, self.worker_id, self.claimed_at = job_id, state, worker_id, claimed_at


class _Mgr:
    def __init__(self, jobs):
        self._lock = threading.Lock()
        self._jobs = OrderedDict((j.job_id, j) for j in jobs)
        self._pending_queue = [j.job_id for j in jobs if j.state == _JobState.PENDING]


def test_force_reclaim_moves_only_remote_claimed_jobs():
    jobs = [
        _Job("a", _JobState.CLAIMED, "worker-1", 1.0),
        _Job("b", _JobState.CLAIMED, "master", 2.0),
        _Job("c", _JobState.PENDING),
        _Job("d", _JobState.COMPLETED, "worker-1"),
        _Job("e", _JobState.CLAIMED, "worker-2", 3.0),
    ]
    mgr = _Mgr(jobs)
    assert go._force_reclaim_remote(mgr, _JobState) == 2
    assert mgr._jobs["a"].state == _JobState.PENDING and mgr._jobs["a"].worker_id is None
    assert mgr._jobs["a"].claimed_at is None
    assert mgr._jobs["e"].state == _JobState.PENDING
    assert mgr._jobs["b"].state == _JobState.CLAIMED and mgr._jobs["b"].worker_id == "master"
    assert mgr._jobs["d"].state == _JobState.COMPLETED
    assert mgr._pending_queue == ["c", "a", "e"]


def test_force_reclaim_with_nothing_to_do():
    assert go._force_reclaim_remote(_Mgr([_Job("a", _JobState.PENDING)]), _JobState) == 0


def test_wait_timeout_default_is_max_of_600_and_claim_timeout():
    assert go._dist_wait_timeout({}) == 600
    assert go._dist_wait_timeout({"claim_timeout": 300}) == 600
    assert go._dist_wait_timeout({"claim_timeout": 1200}) == 1200
    assert go._dist_wait_timeout({"wait_timeout": 30}) == 30
    assert go._dist_wait_timeout({"wait_timeout": "45", "claim_timeout": 1200}) == 45


def test_empty_batch_branch_flushes_waits_and_reclaims():
    with open(os.path.join(ROOT, "generation_orchestrator.py"), encoding="utf-8") as f:
        src = f.read()
    body = re.search(r"\ndef _run_distributed_generation\(.*?(?=\ndef |\Z)", src, re.S).group(0)
    start = body.index("            if not batch:")
    end = body.index("            _idle_since = None\n            _job_t0", start)
    branch = body[start:end]
    assert "_flush_pending_batch(" in branch
    assert branch.index("_flush_pending_batch(") < branch.index("has_pending_or_claimed")
    assert branch.index("has_pending_or_claimed") < branch.index("break")
    assert branch.index("break") < branch.index("_force_reclaim_remote(")
    assert "release_timed_out_jobs()" in branch
    assert "time.sleep(2)" in branch
    assert "continue" in branch
