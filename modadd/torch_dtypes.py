from __future__ import annotations

import torch


DTYPE_LOOKUP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
