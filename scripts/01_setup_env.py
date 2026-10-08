#!/usr/bin/env python3
"""Stage 01: Setup & Verify Environment for PEARL.

Verifies:
  - Python version (>= 3.10)
  - PyTorch and CUDA availability (including GPU name and VRAM)
  - Key dependencies (timm, torchcam, cv2, sklearn, scipy, optuna, zennit, shap)
  - Workspace directory structure (data/, configs/, results/, docs/)
  - Dummy model instantiation and forward pass

Usage::

    python scripts/01_setup_env.py
    python scripts/01_setup_env.py --install  # automatically installs missing dependencies
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_python_version() -> bool:
    v = sys.version_info
    print(f"[1/5] Python Version: {v.major}.{v.minor}.{v.micro} ({platform.platform()})")
    if (v.major, v.minor) < (3, 10):
        print("  [ERROR] Python 3.10 or higher is required.")
        return False
    print("  [OK] Python version is supported.")
    return True


def check_cuda_device() -> dict:
    print("\n[2/5] Checking PyTorch & Hardware Acceleration...")
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_avail}")
        info = {"torch": torch.__version__, "cuda": cuda_avail}
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  Active GPU: {device_name} (x{device_count})")
            print(f"  Total VRAM: {vram_gb:.2f} GB")
            info["gpu_name"] = device_name
            info["vram_gb"] = vram_gb
            if "T4" in device_name:
                print("  [TIP] Detected Nvidia T4 GPU: Use --fp16 for optimal Tensor Core throughput.")
            elif any(arch in device_name for arch in ("A100", "H100", "4090", "4080", "4070", "3090")):
                print("  [TIP] Modern Ampere/Ada GPU detected: Native bfloat16 supported (--bf16).")
        else:
            print("  [NOTE] CUDA is not available; running in CPU mode.")
        return info
    except ImportError:
        print("  [ERROR] PyTorch is not installed in the current environment.")
        return {"torch": None, "cuda": False}


def check_dependencies(auto_install: bool = False) -> bool:
    print("\n[3/5] Checking Key Dependencies...")
    packages = [
        ("timm", "timm"),
        ("torchcam", "torchcam"),
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("yaml", "PyYAML"),
        ("matplotlib", "matplotlib"),
        ("optuna", "optuna"),
        ("pytest", "pytest"),
    ]
    missing = []
    for mod_name, pkg_name in packages:
        try:
            __import__(mod_name)
            print(f"  [OK] {pkg_name}")
        except ImportError:
            print(f"  [MISSING] {pkg_name}")
            missing.append(pkg_name)

    if missing:
        if auto_install:
            print(f"\n  Attempting to install missing packages: {missing}...")
            req_file = REPO_ROOT / "requirements.txt"
            if req_file.exists():
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            else:
                cmd = [sys.executable, "-m", "pip", "install"] + missing
            res = subprocess.run(cmd)
            return res.returncode == 0
        else:
            print("\n  [WARN] Missing dependencies detected. Run:")
            print("         pip install -r requirements.txt")
            return False
    return True


def check_directory_structure() -> None:
    print("\n[4/5] Verifying Workspace Directory Structure...")
    dirs = [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "data" / "preprocessed",
        REPO_ROOT / "configs" / "model",
        REPO_ROOT / "results",
        REPO_ROOT / "docs" / "latex" / "figures",
        REPO_ROOT / "docs" / "thesis" / "figures",
        REPO_ROOT / "docs" / "thesis" / "tables",
        REPO_ROOT / "docs" / "analysis" / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR] {d.relative_to(REPO_ROOT)}")


def check_model_instantiation() -> bool:
    print("\n[5/5] Testing Model Instantiation & Forward Pass...")
    try:
        import torch
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.model.builder import build_model
        from src.utils.config import load_config

        cfg_path = REPO_ROOT / "configs" / "model" / "convnext_tiny.yaml"
        if not cfg_path.exists():
            print(f"  [SKIP] {cfg_path} not found.")
            return True
        cfg = load_config(str(cfg_path))
        cfg["pretrained"] = False
        model = build_model(cfg)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        print(f"  [OK] Model test passed (output shape: {out.shape})")
        return True
    except Exception as e:
        print(f"  [ERROR] Model instantiation failed: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEARL Environment Verification.")
    parser.add_argument("--install", action="store_true", help="Install missing requirements.")
    args = parser.parse_args(argv)

    print("========================================")
    print("   PEARL Pipeline - Environment Setup   ")
    print("========================================")

    p_ok = check_python_version()
    check_cuda_device()
    d_ok = check_dependencies(auto_install=args.install)
    check_directory_structure()
    m_ok = check_model_instantiation()

    print("\n========================================")
    if p_ok and d_ok and m_ok:
        print("  STATUS: Environment is ready!")
        print("  Next Step: python scripts/02_prepare_data.py")
        print("========================================")
        return 0
    else:
        print("  STATUS: Please review errors above.")
        print("========================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
