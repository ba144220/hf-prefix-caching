from typing import Dict, Literal, List
from datetime import datetime
import torch
from transformers.cache_utils import DynamicCache
from hf_prefix_caching.types import CacheHash, CacheBlock
from hf_prefix_caching.utils.cache import crop_first, concat_dynamic_cache
from hf_prefix_caching.utils.hashing import batched_hash

INIT_HASH = CacheHash("0")

class PrefixCache:
    def __init__(
        self,
        max_num_blocks: int = 2048,
        max_block_length: int = 100, # i.e. total length is max_block_length * block_size
        block_size: int = 16,
        padding_side: Literal["left", "right"] = "right",
    ):
        self.max_num_blocks = max_num_blocks
        self.max_block_length = max_block_length
        self.block_size = block_size
        self.padding_side = padding_side
        if padding_side == "left":
            raise NotImplementedError("Left padding is not supported yet")

        self.caches: Dict[CacheHash, CacheBlock] = {}

    def update(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
    ):
        # Check past_key_values is a DynamicCache
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError("past_key_values must be a DynamicCache")
        
        # Check if input_ids and past_key_values have the same batch size
        if input_ids.shape[0] != past_key_values[0][0].shape[0]:
            raise ValueError("input_ids and past_key_values must have the same batch size")
        
        # Crop input_ids to the multiple of block_size
        batch_size, seq_len = input_ids.shape
        block_num = min(seq_len // self.block_size, self.max_block_length)
        input_ids = input_ids[:, :block_num * self.block_size]

        # Crop past_key_values to the multiple of block_size
        past_key_values.crop(block_num * self.block_size)

        prev_hashes = [INIT_HASH for _ in range(batch_size)]

        for block_idx in range(block_num):
            start_pos = block_idx * self.block_size
            end_pos = start_pos + self.block_size

            new_hashes: List[CacheHash] = batched_hash(prev_hashes, input_ids[:, start_pos:end_pos])
            cropped_past_key_values = crop_first(past_key_values, self.block_size)
            kv_caches: List[DynamicCache] = cropped_past_key_values.batch_split(batch_size, 1)

            for batch_idx in range(batch_size):
                block_hash = new_hashes[batch_idx]
                if block_hash not in self.caches:
                    if len(self.caches) >= self.max_num_blocks:
                        self.caches.pop(min(self.caches.keys(), key=lambda x: (self.caches[x].num_hits, self.caches[x].created_at)))

                    self.caches[block_hash] = CacheBlock(
                        cache=kv_caches[batch_idx],
                        num_hits=1,
                        created_at=datetime.now(),
                        start_pos=start_pos,
                        end_pos=end_pos,
                        input_ids=input_ids[batch_idx, start_pos:end_pos].tolist(),
                    )
                else:
                    self.caches[block_hash].num_hits += 1

                prev_hashes[batch_idx] = block_hash    
    def get(
        self,
        input_ids: torch.Tensor,
    ) -> DynamicCache:
        # Crop input_ids to the multiple of block_size
        batch_size, seq_len = input_ids.shape
        block_num = seq_len // self.block_size
        input_ids = input_ids[:, :block_num * self.block_size]

        prev_hashes = [INIT_HASH for _ in range(batch_size)]

        caches: List[DynamicCache] = []

        for block_idx in range(block_num):
            start_pos = block_idx * self.block_size
            end_pos = start_pos + self.block_size

            new_hashes: List[CacheHash] = batched_hash(prev_hashes, input_ids[:, start_pos:end_pos])
            batch_caches: List[DynamicCache] = []
            for batch_idx in range(batch_size):
                block_hash = new_hashes[batch_idx]
                
                if block_hash in self.caches:
                    batch_caches.append(self.caches[block_hash].cache)
                else:
                    break

                prev_hashes[batch_idx] = block_hash

            if len(batch_caches) != batch_size:
                break
            caches.append(DynamicCache.from_batch_splits(batch_caches))

        return concat_dynamic_cache(caches)

    def get_num_blocks(self) -> int:
        return len(self.caches)