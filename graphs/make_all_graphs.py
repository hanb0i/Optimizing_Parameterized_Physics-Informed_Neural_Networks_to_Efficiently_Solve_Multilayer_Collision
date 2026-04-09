from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "graphs" / "scripts"


def _python() -> str:
    return sys.executable or "python3"


def _run(script: str) -> None:
    script_path = SCRIPTS_DIR / script
    print(f"\n=== {script_path} ===")
    subprocess.run([_python(), str(script_path)], cwd=str(REPO_ROOT), check=False)


def main() -> None:
    _run("plot_geometry_bc.py")
    _run("plot_ablation_table.py")
    _run("plot_error_heatmap.py")
    _run("plot_surrogate_verification.py")
    print("\nDone. Figures are in `graphs/figures/`.")


if __name__ == "__main__":
    main()

