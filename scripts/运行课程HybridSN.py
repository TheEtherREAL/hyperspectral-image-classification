"""Run one YAML-defined HybridSN coursework experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coursework.stage2 import run_stage2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = run_stage2(PROJECT_ROOT, config, output, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
