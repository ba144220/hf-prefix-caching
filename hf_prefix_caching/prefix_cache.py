from typing import Dict, Literal, List
from datetime import datetime
import torch
from transformers.cache_utils import DynamicCache
from .types import CacheHash, CacheBlock
from .utils.cache import crop_first, concat_dynamic_cache
from .utils.hashing import batched_hash

INIT_HASH = CacheHash("0")

class PrefixCache:
    def __init__(
        self,
        max_num_blocks: int = 2048,
        max_block_length: int = 100, # i.e. total length is max_block_length * block_size
        max_new_blocks: int = 4, # max number of new blocks to add in one update
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

        self.epoch = 0
        self.evict_size = 0.1
        self.max_new_blocks = max_new_blocks

    def _get_batch_eviction_candidates(self, num_to_evict: int) -> List[CacheHash]:
        """Get multiple eviction candidates sorted by priority (lowest first)"""
        if not self.caches or num_to_evict <= 0:
            return []
            
        current_epoch = self.epoch
        
        # Calculate stats for normalization
        all_blocks = list(self.caches.values())
        max_hits = max(b.num_hits for b in all_blocks)
        max_age = max(current_epoch - b.created_at for b in all_blocks) or 1
        
        def calculate_priority_score(block: CacheBlock) -> float:
            # Higher score = higher priority = keep longer
            
            # Hit frequency (normalized)
            hit_score = block.num_hits / max_hits if max_hits > 0 else 0
            
            # Recency (recent access is valuable)
            epochs_since_access = current_epoch - block.last_hit_at
            recency_score = 1.0 / (1 + epochs_since_access)
            
            # Age penalty (very old blocks are less valuable)
            age = current_epoch - block.created_at
            age_score = 1.0 / (1 + age / max_age)
            
            # Position value (earlier sequence positions often more important)
            position_score = 1.0 / (1 + block.start_pos / 1000)
            
            # Weighted combination
            return (0.2 * hit_score + 
                    0.2 * recency_score + 
                    0.2 * age_score + 
                    0.4 * position_score)
        
        # Sort all cache entries by priority (lowest first = evict first)
        sorted_entries = sorted(
            self.caches.keys(),
            key=lambda x: calculate_priority_score(self.caches[x])
        )
        
        # Return the lowest priority entries up to num_to_evict
        return sorted_entries[:min(num_to_evict, len(sorted_entries))]

    def _batch_evict(self, num_to_evict: int) -> int:
        """Evict multiple blocks at once. Returns number of blocks actually evicted."""
        candidates = self._get_batch_eviction_candidates(num_to_evict)
        
        evicted_count = 0
        for block_hash in candidates:
            if block_hash in self.caches:  # Double-check it still exists
                self.caches.pop(block_hash)
                evicted_count += 1
        return evicted_count
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

        new_blocks = [0] * batch_size

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
                        self._batch_evict(int(self.max_num_blocks * self.evict_size))

                    if new_blocks[batch_idx] < self.max_new_blocks:
                        self.caches[block_hash] = CacheBlock(
                            cache=kv_caches[batch_idx],
                            num_hits=1,
                            created_at=self.epoch,
                            last_hit_at=self.epoch,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            input_ids=input_ids[batch_idx, start_pos:end_pos].tolist(),
                        )
                        new_blocks[batch_idx] += 1
                else:
                    self.caches[block_hash].num_hits += 1
                    self.caches[block_hash].last_hit_at = self.epoch

                prev_hashes[batch_idx] = block_hash

        self.epoch += 1

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