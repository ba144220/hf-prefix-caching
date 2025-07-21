import copy
from transformers.cache_utils import DynamicCache

def crop_dynamic_cache(
    cache: DynamicCache,
    start_idx: int,
    end_idx: int,
) -> DynamicCache:
    new_cache = copy.deepcopy(cache)
    for idx in range(len(new_cache.key_cache)):
        if new_cache.key_cache[idx].numel():
            new_cache.key_cache[idx] = new_cache.key_cache[idx][..., start_idx:end_idx, :]
            new_cache.value_cache[idx] = new_cache.value_cache[idx][..., start_idx:end_idx, :]
    return new_cache


if __name__ == "__main__":
    import torch
    cache = DynamicCache()

    x = torch.randn(1, 4, 10, 2)
    cache.key_cache = [x for _ in range(10)]
    cache.value_cache = [x for _ in range(10)]
    print(cache.key_cache[0].shape)
    new_cache = crop_dynamic_cache(cache, 0, 5)
    print(new_cache.key_cache[0].shape)
    print(cache.key_cache[0].shape)

    print(torch.allclose(x[..., 0:5, :], crop_dynamic_cache(cache, 0, 5).key_cache[0]))