"""Run the reusable stage-1 coursework preprocessing workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coursework.stage1 import run_stage1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "coursework/configs/stage1_data/pavia_university.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "coursework/outputs/stage1/pavia_university",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    result = run_stage1(
        PROJECT_ROOT,
        args.config,
        args.output_dir,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
