"""Execute the three coursework notebooks and fail on the first cell error."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "coursework/notebooks"
KERNEL_ROOT = PROJECT_ROOT / "coursework/jupyter_runtime/share/jupyter"


def main() -> None:
    os.environ["JUPYTER_PATH"] = str(KERNEL_ROOT)
    os.environ["JUPYTER_CONFIG_DIR"] = str(PROJECT_ROOT / "coursework/.jupyter_config")
    os.environ["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / "coursework/.jupyter_data")
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="hsi-coursework",
            allow_errors=False,
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        )
        client.execute(cwd=str(NOTEBOOK_DIR))
        nbformat.write(notebook, path)
        print(f"executed: {path.name}")


if __name__ == "__main__":
    main()
