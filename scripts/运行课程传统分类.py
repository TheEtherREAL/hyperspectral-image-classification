"""Run all YAML-defined traditional coursework comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coursework.stage3 import run_all_traditional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "coursework/configs/stage3_traditional/pavia_traditional.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "coursework/outputs/stage3/pavia_traditional",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    rows = run_all_traditional(PROJECT_ROOT, config, output)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
