from dataclasses import dataclass
from typing import NewType
from datetime import datetime
import torch
from transformers.cache_utils import Cache

@dataclass
class CacheBlock:
    cache: Cache
    position_ids: torch.Tensor
    num_hits: int
    created_at: datetime

CacheHash = NewType("CacheHash", str)

