from typing import List
import torch
import hashlib
from ..types import CacheHash

def batched_hash(
    prev_hash: List[CacheHash],
    input_ids: torch.Tensor,
) -> List[CacheHash]:
    """
    Hash the input_ids and the previous hash.
        new_hash = hash(prev_hash + input_ids)
    """
    if len(prev_hash) != input_ids.shape[0]:
        raise ValueError(
            f"Length of prev_hash ({len(prev_hash)}) must match input_ids.shape[0] ({input_ids.shape[0]})"
        )
    
    new_hash = []
    for i in range(input_ids.shape[0]):
        input_ids_string = str(input_ids[i].tolist())
        new_hash.append(hashlib.sha256(
            (prev_hash[i] + input_ids_string).encode()
        ).hexdigest())

    return new_hash

if __name__ == "__main__":
    prev_hash = [CacheHash("123"), CacheHash("456")]
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    new_hash = batched_hash(prev_hash, input_ids)
    print(new_hash)