"""Resolve project root whether cwd is repo root or notebooks/."""

import sys
from pathlib import Path


def setup_project_path():
    cwd = Path.cwd().resolve()
    if (cwd / "src").is_dir() and (cwd / "config").is_dir():
        root = cwd
    elif (cwd.parent / "src").is_dir():
        root = cwd.parent
    else:
        root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
