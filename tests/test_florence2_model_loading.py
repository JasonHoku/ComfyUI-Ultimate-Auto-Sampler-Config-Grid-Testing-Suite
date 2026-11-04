"""Tests for Florence2 model loading + caching."""
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.requires_torch


def test_load_florence2_model_caches_per_name(monkeypatch):
    """Same model name -> load called once."""
    load_call_count = {"n": 0}

    class FakeLoader:
        FUNCTION = "loadmodel"

        def loadmodel(self, **kwargs):
            load_call_count["n"] += 1
            return (f"loaded:{kwargs['model']}",)

    import nodes
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "DownloadAndLoadFlorence2Model": FakeLoader,
        "Florence2Run": object,
    }, raising=False)

    # Reset cache (module-level dict)
    import florence2_hires
    florence2_hires._FLORENCE2_MODEL_CACHE.clear()

    m1 = florence2_hires.load_florence2_model("microsoft/Florence-2-base")
    m2 = florence2_hires.load_florence2_model("microsoft/Florence-2-base")
    assert m1 is m2
    assert load_call_count["n"] == 1


def test_load_florence2_model_different_names_load_separately(monkeypatch):
    load_call_count = {"n": 0}

    class FakeLoader:
        FUNCTION = "loadmodel"

        def loadmodel(self, **kwargs):
            load_call_count["n"] += 1
            return (f"loaded:{kwargs['model']}",)

    import nodes
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "DownloadAndLoadFlorence2Model": FakeLoader,
        "Florence2Run": object,
    }, raising=False)

    import florence2_hires
    florence2_hires._FLORENCE2_MODEL_CACHE.clear()

    m1 = florence2_hires.load_florence2_model("microsoft/Florence-2-base")
    m2 = florence2_hires.load_florence2_model("microsoft/Florence-2-large")
    assert m1 != m2
    assert load_call_count["n"] == 2


def test_load_florence2_passes_hidden_defaults(monkeypatch):
    """Verify hidden defaults: precision=fp16, attention=sdpa, convert_to_safetensors=False."""
    captured = {}

    class FakeLoader:
        FUNCTION = "loadmodel"

        def loadmodel(self, **kwargs):
            captured.update(kwargs)
            return ("model_handle",)

    import nodes
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "DownloadAndLoadFlorence2Model": FakeLoader,
        "Florence2Run": object,
    }, raising=False)

    import florence2_hires
    florence2_hires._FLORENCE2_MODEL_CACHE.clear()
    florence2_hires.load_florence2_model("microsoft/Florence-2-base")

    assert captured["model"] == "microsoft/Florence-2-base"
    assert captured["precision"] == "fp16"
    assert captured["attention"] == "sdpa"
    assert captured["convert_to_safetensors"] is False
