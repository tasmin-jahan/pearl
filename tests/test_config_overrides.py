"""
Pytest suite — dynamic CLI config overrides.
"""

from src.utils.config import parse_overrides, apply_overrides


def test_parse_overrides_basic():
    result = parse_overrides(["training.lr=5e-4", "training.batch_size=16"])
    assert result == {"training": {"lr": 5e-4, "batch_size": 16}}


def test_parse_overrides_types():
    result = parse_overrides([
        "training.lr=1.5e-4",
        "training.epochs=100",
        "training.bf16=true",
        "name=foo",
    ])
    assert result["training"]["lr"] == 1.5e-4
    assert result["training"]["epochs"] == 100
    assert result["training"]["bf16"] is True
    assert result["name"] == "foo"


def test_apply_overrides_deep_merge():
    base = {"training": {"lr": 1e-4, "batch_size": 32, "warmup_epochs": 2}}
    overrides = {"training": {"lr": 5e-4}}
    merged = apply_overrides(base, overrides)
    assert merged["training"]["lr"] == 5e-4
    assert merged["training"]["batch_size"] == 32  # untouched
    assert merged["training"]["warmup_epochs"] == 2