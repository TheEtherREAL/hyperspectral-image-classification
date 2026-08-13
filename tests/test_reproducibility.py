import random

import numpy as np

from src.utils.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_numpy() -> None:
    seed_everything(42)
    first = (random.random(), np.random.rand())

    seed_everything(42)
    second = (random.random(), np.random.rand())

    assert first == second

