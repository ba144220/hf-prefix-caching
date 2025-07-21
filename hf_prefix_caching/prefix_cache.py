from typing import Dict, Literal
import torch
from hf_prefix_caching.types import CacheHash, CacheBlock

class PrefixCache:
    def __init__(
        self,
        max_num_blocks: int = 2048,
        block_size: int = 16,
        padding_side: Literal["left", "right"] = "right",
    ):
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.padding_side = padding_side
        if padding_side == "left":
            raise NotImplementedError("Left padding is not supported yet")

        self.caches: Dict[CacheHash, CacheBlock] = {}

    
    
