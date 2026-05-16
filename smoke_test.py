from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    cfile = Path(tempfile.gettempdir()) / "virtual_mouse_keyboard_main.pyc"
    py_compile.compile(str(ROOT / "main.py"), cfile=str(cfile), doraise=True)
    print("Smoke test passed: main.py compiles.")


if __name__ == "__main__":
    main()
