from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(7)
