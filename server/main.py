"""Run the VeloxDB FastAPI server: python -m server.main"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import uvicorn

from server.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
