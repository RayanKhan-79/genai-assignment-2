"""Tests for training artefact helpers (no model is trained here)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_train_curve_plot_overrides_colab_backend(tmp_path: Path) -> None:
    """Colab exports a matplotlib_inline backend the uv venv lacks; plotting must still work.

    The check runs in a fresh interpreter because matplotlib caches its backend at import time.
    """
    script = (
        "import os, sys;"
        "os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline';"
        "from pathlib import Path;"
        "from trialmatch.train import TrainCurvePlotter;"
        "path = Path(sys.argv[1]);"
        "TrainCurvePlotter.plot("
        "[{'step': 1, 'loss': 1.0}, {'step': 5, 'loss': 0.5}, {'step': 5, 'eval_loss': 0.6}],"
        "path)"
    )
    path = tmp_path / "curves.pdf"
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        timeout=180,
        env={**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"},
    )
    assert path.read_bytes().startswith(b"%PDF")
