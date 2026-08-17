from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coursework.preprocessing_benchmark import run_preprocessing_svm_benchmark


if __name__ == "__main__":
    run_preprocessing_svm_benchmark(
        PROJECT_ROOT,
        PROJECT_ROOT / "coursework/outputs/stage1/pavia_university/stage1_manifest.json",
        PROJECT_ROOT / "coursework/outputs/comparisons/preprocessing_svm",
    )
