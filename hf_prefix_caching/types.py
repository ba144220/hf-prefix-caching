from dataclasses import dataclass
from typing import NewType, Optional
from datetime import datetime
import torch
from transformers.cache_utils import DynamicCache

@dataclass
class CacheBlock:
    cache: DynamicCache
    num_hits: int
    created_at: datetime
    start_pos: int
    end_pos: int # exclusive
    input_ids: Optional[torch.Tensor] = None

    def __str__(self):
        return f"CacheBlock(num_hits={self.num_hits}, created_at={self.created_at}, start_pos={self.start_pos}, end_pos={self.end_pos}, input_ids={self.input_ids})"

CacheHash = NewType("CacheHash", str)

