import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WICKIT_ROOT = REPO_ROOT / "external" / "wickit"

if str(WICKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(WICKIT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "TRUE")
