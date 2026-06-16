import sys
from pathlib import Path

# Ensure repo root is on sys.path so all layer packages are importable
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
