"""A connected latent's pixel size is cells x the model's latent ratio, not cells * 8."""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from config_utils import prepare_input_jobs, latent_job_pixel_size  # noqa: E402


class T:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, i):
        return T(self.shape[1:])

    def unsqueeze(self, d):
        return T((1,) + self.shape)


def test_prepare_input_jobs_carries_cells():
    jobs = prepare_input_jobs({"samples": T((2, 128, 76, 52))}, [])
    assert len(jobs) == 2
    assert [j["batch_idx"] for j in jobs] == [0, 1]
    assert all(j["latent_cells"] == (52, 76) for j in jobs)
    assert (jobs[0]["width"], jobs[0]["height"]) == (416, 608)
    assert prepare_input_jobs({"samples": T((1, 4, 8, 8))}, [], 16)[0]["width"] == 128


def test_latent_job_pixel_size():
    j = prepare_input_jobs({"samples": T((1, 128, 76, 52))}, [])[0]
    assert latent_job_pixel_size(j, 16) == (832, 1216)
    assert latent_job_pixel_size(j, 8) == (416, 608)
    assert latent_job_pixel_size(j, None) == (416, 608)
    r = prepare_input_jobs(None, [[512, 768]])[0]
    assert latent_job_pixel_size(r, 16) == (512, 768)


def _read(n):
    with open(os.path.join(_ROOT, n), encoding="utf-8") as f:
        return f.read()


def test_no_hardcoded_8_in_prepare_input_jobs():
    body = re.search(r"def prepare_input_jobs\(.*?\ndef ", _read("config_utils.py"), re.S).group(0)
    assert "* 8" not in body


def test_orchestrator_resolves_size_after_ratio():
    s = _read("generation_orchestrator.py")
    a = s.index("latent_ratio = get_latent_downscale_ratio(loaded_model)")
    b = s.index("latent_job_pixel_size(job, latent_ratio)")
    c = s.index("make_empty_latent(latent_channels, gen_w")
    assert a < b < c
    # distributed master too
    assert 'optional_latent["samples"].shape[3]) * latent_ratio' in s
