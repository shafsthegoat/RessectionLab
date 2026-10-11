"""Staging-only bootstrap; the utility tests themselves use canonical imports."""
from pathlib import Path
import sys
import resectionlab

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parents[1]
resectionlab.__path__.insert(0, str(STAGE / "stage/resectionlab"))
sys.path.insert(0, str(ROOT / "build/hybrid-fixed-four-switch-regression-v1"))
sys.path.insert(0, str(STAGE))
