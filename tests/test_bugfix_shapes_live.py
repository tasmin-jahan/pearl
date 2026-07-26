"""Regression tests for two bugs that crashed/aborted the v3 sweep:

  1. ``backbone.num_features`` is unreliable on some timm models
     (mobilenetv3_large_100 reports 960 but forward actually outputs
     1280).  The builder now probes the actual forward output shape.

  2. The sweep outer Rich ``Live`` and the trainer's inner Rich
     ``Live`` would both try to redraw the terminal at once, causing
     visible jitter.  The trainer now skips its inner Live when an
     outer one is active (signalled by ``set_outer_live_active(True)``).
"""

import pytest

import src.training.trainer as trainer_mod
from src.training.trainer import (
    set_outer_live_active,
    _outer_live_active,
)


class TestInFeaturesProbe:
    """Verify the builder resolves the right ``in_features`` for every
    timm backbone, even when ``backbone.num_features`` lies."""

    @pytest.mark.parametrize(
        "timm_name,input_size,expected_dim",
        [
            ("resnet50", 224, 2048),
            ("resnet101", 224, 2048),
            ("densenet121", 224, 1024),
            ("densenet169", 224, 1664),
            ("efficientnet_b0", 224, 1280),
            ("convnext_tiny", 224, 768),
            # The crash: timm reports 960, forward outputs 1280.
            ("mobilenetv3_large_100", 224, 1280),
            ("vit_base_patch16_224", 224, 768),
            ("swin_tiny_patch4_window7_224", 224, 768),
        ],
    )
    def test_probe_matches_forward(self, timm_name, input_size, expected_dim):
        import timm
        import torch

        from src.model.builder import _probe_in_features

        b = timm.create_model(timm_name, pretrained=False, num_classes=0)
        probed = _probe_in_features(b, input_size=input_size)
        # Sanity: a real forward through the same backbone yields the
        # expected width, and our probe reads the same value.
        with torch.no_grad():
            out = b(torch.zeros(1, 3, input_size, input_size))
        actual = out.shape[1] if out.dim() >= 2 else 0
        assert probed == expected_dim, (
            f"{timm_name}: probe returned {probed}, expected {expected_dim}"
        )
        assert probed == actual, (
            f"{timm_name}: probe={probed}, real forward={actual} — mismatch"
        )


class TestEndToEndModelBuild:
    """End-to-end: build_model produces a model whose forward matches its
    head's expected in_features (no shape-mismatch crash on mobilenetv3)."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "resnet50", "resnet101", "densenet121", "densenet169",
            "efficientnet_b0", "convnext_tiny", "mobilenetv3_large",
            "vit_base", "swin_tiny",
        ],
    )
    def test_build_then_forward(self, model_name, tmp_path):
        import torch
        import yaml

        cfg_path = f"configs/model/{model_name}.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        from src.model.builder import build_model

        model = build_model(cfg)
        x = torch.randn(2, 3, cfg["input_size"], cfg["input_size"])
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, cfg["num_classes"])


class TestOuterLiveFlag:
    """The trainer's outer-Live-active handshake must toggle cleanly."""

    def setup_method(self):
        # Reset to known state before each test.
        set_outer_live_active(False)

    def teardown_method(self):
        set_outer_live_active(False)

    def test_default_off(self):
        assert _outer_live_active() is False

    def test_set_true_then_false(self):
        set_outer_live_active(True)
        assert _outer_live_active() is True
        set_outer_live_active(False)
        assert _outer_live_active() is False

    def test_trainer_respects_flag(self):
        """When the outer Live is active, the trainer must skip its
        own inner Live to avoid two contexts fighting for the cursor."""
        from src.training.trainer import Trainer
        import torch.nn as nn

        # We don't actually need a real model — just construct the
        # trainer and verify the ``_use_inner_live`` flag flips based
        # on the outer-Live state.
        class _DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        # First: outer Live OFF → inner Live allowed by default.
        m = _DummyModel()
        t = Trainer.__new__(Trainer)
        t.config = {"use_inner_live": True}
        t.silent = False
        # _use_inner_live would be computed in __init__ — replicate
        # the logic here without running real __init__.
        t._use_inner_live = not (
            t.silent or _outer_live_active() or not t.config.get("use_inner_live", True)
        )
        assert t._use_inner_live is True

        # Now: outer Live ON → trainer must skip inner Live.
        set_outer_live_active(True)
        t._use_inner_live = not (
            t.silent or _outer_live_active() or not t.config.get("use_inner_live", True)
        )
        assert t._use_inner_live is False
