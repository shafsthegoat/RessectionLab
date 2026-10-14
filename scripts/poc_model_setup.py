#!/usr/bin/env python3
"""Create a separate POC inference environment; never alter the app environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".tools/poc-model-runtime/venv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    if not (RUNTIME / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(RUNTIME)], check=True)
        subprocess.run([str(RUNTIME / "bin/python"), "-m", "pip", "install", "-r",
                        str(ROOT / "manifests/poc-model-runtime-requirements.txt")], check=True)
    subprocess.run([str(RUNTIME / "bin/python"), "-c",
                    "import torch,monai,totalsegmentator,nnunetv2,nibabel,SimpleITK,surfa; print('POC runtime import check passed')"], check=True)
    if args.download_weights:
        subprocess.run([str(RUNTIME / "bin/python"), str(ROOT / "scripts/poc_model_assets.py")], check=True)


if __name__ == "__main__":
    main()
