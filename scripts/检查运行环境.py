"""检查项目 Python、依赖与 GPU / Check software and accelerator state."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    try:
        import numpy as np

        print(f"numpy={np.__version__}")
    except ImportError:
        print("numpy=missing")

    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"device_name={torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch=missing")


if __name__ == "__main__":
    main()
