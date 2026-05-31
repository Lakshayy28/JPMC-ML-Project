from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    uvicorn.run(app="fri.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()