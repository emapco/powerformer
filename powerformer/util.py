import importlib.util
import sys
from typing import Literal

import torch
from transformers.utils import is_torch_npu_available


def get_device_name() -> Literal["cuda", "mps", "npu", "hpu", "cpu"]:
    """
    Returns the name of the device where this module is running on.

    It's a simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly.

    Returns:
        str: Device name, like 'cuda' or 'cpu'
    """
    if "debugpy" in sys.modules:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    elif importlib.util.find_spec("habana_frameworks") is not None:
        import habana_frameworks.torch.hpu as hthpu  # type: ignore

        if hthpu.is_available():
            return "hpu"
    return "cpu"
