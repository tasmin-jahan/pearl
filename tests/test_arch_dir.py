"""Tests for the per-arch directory helpers in src/utils/logging.py."""

import os
import tempfile

import pytest

from src.utils.logging import (
    ExperimentLogger,
    external_validation_path,
    make_arch_dir,
    make_run_dir,
)


class TestMakeArchDir:
    """The per-(prep, arch) directory builder."""

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            arch_dir = make_arch_dir(tmp, "srad", "efficientnet_b0")
            assert arch_dir == os.path.join(tmp, "checkpoints", "srad",
                                            "efficientnet_b0")
            assert os.path.isdir(arch_dir)

    def test_idempotent(self):
        """Calling twice doesn't fail (makedirs exist_ok=True)."""
        with tempfile.TemporaryDirectory() as tmp:
            a = make_arch_dir(tmp, "srad", "efficientnet_b0")
            b = make_arch_dir(tmp, "srad", "efficientnet_b0")
            assert a == b
            assert os.path.isdir(a)

    def test_separate_prep_dirs(self):
        """Different prep names land in different <prep>/<arch>/ subtrees."""
        with tempfile.TemporaryDirectory() as tmp:
            a = make_arch_dir(tmp, "srad", "eff")
            b = make_arch_dir(tmp, "gauss", "eff")
            assert a != b
            # Same arch name in different prep directories.
            assert a.endswith("/checkpoints/srad/eff")
            assert b.endswith("/checkpoints/gauss/eff")
            assert os.path.isdir(a)
            assert os.path.isdir(b)

    def test_separate_arch_dirs(self):
        """Different arch names land as siblings under the same <prep>/."""
        with tempfile.TemporaryDirectory() as tmp:
            a = make_arch_dir(tmp, "srad", "resnet50")
            b = make_arch_dir(tmp, "srad", "efficientnet_b0")
            assert os.path.dirname(a) == os.path.dirname(b)
            assert a != b


class TestExternalValidationPath:
    """The external-validation sub-path resolver."""

    def test_default_dataset_and_ext(self):
        path = external_validation_path("/tmp/x/eff_b0")
        assert path == "/tmp/x/eff_b0/external_validation/pcosgen.json"

    def test_custom_dataset(self):
        path = external_validation_path(
            "/tmp/x/eff_b0", dataset="Mendeley", ext="csv",
        )
        assert path == "/tmp/x/eff_b0/external_validation/Mendeley.csv"

    def test_under_make_arch_dir(self):
        """The two helpers compose — path is a real, creatable location."""
        with tempfile.TemporaryDirectory() as tmp:
            arch_dir = make_arch_dir(tmp, "srad", "eff")
            path = external_validation_path(arch_dir, "pcosgen", "json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            assert os.path.isdir(os.path.dirname(path))


class TestBackwardCompat:
    """make_run_dir still works for legacy callers."""

    def test_make_run_dir_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_run_dir(tmp, "swin_tiny", "srad")
            # Format: <results>/runs/<arch>__<prep>__<ts>/
            assert "runs" in d
            assert "swin_tiny__srad__" in d


class TestExperimentLoggerInArchDir:
    """ExperimentLogger works with the per-arch dir as run_dir."""

    def test_logger_writes_inside_arch_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            arch_dir = make_arch_dir(tmp, "srad", "eff")
            cfg = {"model": {"name": "eff"}, "preprocessing": {"name": "srad"}}
            logger = ExperimentLogger(arch_dir, cfg)
            try:
                assert logger.run_dir == arch_dir
                assert os.path.isfile(os.path.join(arch_dir, "config.yaml"))
                assert os.path.isfile(os.path.join(arch_dir, "epoch_log.csv"))
            finally:
                logger.close()
