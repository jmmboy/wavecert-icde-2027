from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
panel_dir = os.environ.get("WAVECERT_PANEL_DIR")
PANELS = (Path(panel_dir).expanduser() if panel_dir else ROOT / "private" / "panels").resolve()

raw_dir = os.environ.get("WAVECERT_RAW_DIR")
RAW_DATA = (Path(raw_dir).expanduser() if raw_dir else ROOT / "private" / "raw").resolve()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
