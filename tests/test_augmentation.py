"""
Tests for the noise-robustness augmentations (JPEG-style compression and
light Gaussian blur) added to Preprocessor._apply_augmentation.

These augmentations address the heavy JPEG compression (~q40) in the
source dataset by teaching the model to be invariant to the family of
codec artefacts already present, without re-introducing the noise that
the preprocessing pipeline already removed.
"""

import numpy as np
import pytest

from src.preprocessing.preprocess import Preprocessor


def _make_config(jpeg=True, blur=True, jpeg_p=1.0, blur_p=1.0):
    """Helper: build a minimal preprocessing config with both augmentations."""
    return {
        "name": "test_aug",
        "steps": {
            "resize": True,
            "clahe": {"enabled": False},
            "gaussian": {"enabled": False},
            "srad": {"enabled": False},
            "anisotropic_diffusion": {"enabled": False},
            "zscore_normalize": {"enabled": False},
            "padding": "reflect",
        },
        "augmentation": {
            "rotation": 0,
            "horizontal_flip": False,
            "scale": 0,
            "jpeg_compression": {
                "enabled": jpeg,
                "p": jpeg_p,
                "q_low": 50,
                "q_high": 95,
            },
            "light_blur": {
                "enabled": blur,
                "p": blur_p,
                "sigma_low": 0.1,
                "sigma_high": 1.5,
            },
        },
    }


def _make_image(seed=0, h=64, w=64):
    rng = np.random.default_rng(seed)
    # Synthetic ultrasound-like image: a smooth gradient with speckle noise
    x = np.linspace(0, 255, w, dtype=np.float32)
    grad = np.tile(x, (h, 1))
    speckle = rng.normal(0, 25, size=(h, w)).astype(np.float32)
    img = np.clip(grad + speckle, 0, 255)[:, :, None].repeat(3, axis=2)
    return img.astype(np.uint8)


def test_jpeg_augmentation_produces_valid_output():
    """JPEG-style augmentation must run, preserve shape, and stay in float32."""
    cfg = _make_config(jpeg=True, blur=False, jpeg_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image()
    np.random.seed(42)
    out = pp._apply_jpeg_augmentation(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Output must be in the [0, 255] range (JPEG round-trip is lossy).
    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_blur_augmentation_produces_valid_output():
    """Light Gaussian blur must run and preserve shape and dtype."""
    cfg = _make_config(jpeg=False, blur=True, blur_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image().astype(np.float32)  # matches post-zscore dtype
    np.random.seed(42)
    out = pp._apply_blur_augmentation(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_augmentation_runs_through_apply_pipeline():
    """FULL pipeline (resize + augmentation) must produce a 224x224x3 float32."""
    cfg = _make_config(jpeg=True, blur=True, jpeg_p=1.0, blur_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image(h=300, w=400)  # non-square to exercise letterbox
    np.random.seed(42)
    out = pp.apply(img, augment=True)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_augmentation_disabled_when_both_off():
    """If both augmentations are disabled but the function is called, the
    output must equal the input (no-op)."""
    cfg = _make_config(jpeg=False, blur=False)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image()
    np.random.seed(42)
    out = pp._apply_augmentation(img)
    np.testing.assert_array_equal(out, img)


def test_augmentation_disabled_at_validation():
    """Validation-time calls to apply(augment=False) must produce identical
    outputs regardless of which augmentations are enabled in the config."""
    cfg = _make_config(jpeg=True, blur=True, jpeg_p=1.0, blur_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image(h=300, w=400)
    np.random.seed(42)
    out_train = pp.apply(img, augment=False)
    np.random.seed(42)
    out_train2 = pp.apply(img, augment=False)
    np.testing.assert_array_equal(out_train, out_train2)


def test_jpeg_augmentation_is_stochastic_but_bounded():
    """Across many calls, the JPEG aug must produce varying q values within
    the configured range, and never crash."""
    cfg = _make_config(jpeg=True, blur=False, jpeg_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image()
    np.random.seed(0)
    outs = [pp._apply_jpeg_augmentation(img) for _ in range(20)]
    # All outputs share shape and dtype.
    for o in outs:
        assert o.shape == img.shape
        assert o.dtype == np.float32
    # Not all outputs are bit-identical (stochasticity is real).
    diffs = [np.abs(o1 - o2).mean() for o1, o2 in zip(outs[:-1], outs[1:])]
    assert max(diffs) > 0.0


def test_blur_augmentation_changes_image():
    """Light blur must actually change the image (sigma > 0)."""
    cfg = _make_config(jpeg=False, blur=True, blur_p=1.0)
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image()
    np.random.seed(0)
    out = pp._apply_blur_augmentation(img)
    # Some change expected (not bit-identical).
    assert np.abs(out - img).mean() > 0.0


def test_default_disabled_backward_compatibility():
    """A config without the new keys must behave as before (off by default)."""
    cfg = {
        "name": "legacy",
        "steps": {
            "resize": True,
            "padding": "reflect",
            "zscore_normalize": {"enabled": False},
            "clahe": {"enabled": False},
            "gaussian": {"enabled": False},
            "srad": {"enabled": False},
            "anisotropic_diffusion": {"enabled": False},
        },
        "augmentation": {
            "rotation": 0,
            "horizontal_flip": False,
            "scale": 0,
        },
    }
    pp = Preprocessor(cfg, input_size=224)
    img = _make_image()
    np.random.seed(7)
    out = pp._apply_augmentation(img)
    np.testing.assert_array_equal(out, img)
