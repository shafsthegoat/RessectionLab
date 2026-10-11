from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests'))
import resectionlab
resectionlab.__path__.insert(0, str(Path(__file__).resolve().parents[1]/'stage/resectionlab'))
