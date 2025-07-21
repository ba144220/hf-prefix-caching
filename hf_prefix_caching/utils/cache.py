import copy
from typing import List
import torch
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

def concat_dynamic_cache(
    caches: List[DynamicCache],
) -> DynamicCache:
    new_cache = DynamicCache()
    if len(caches) == 0:
        return new_cache
    for idx in range(len(caches[0])):
        key_cache = [current.key_cache[idx] for current in caches if current.key_cache[idx].numel()]
        value_cache = [current.value_cache[idx] for current in caches if current.value_cache[idx].numel()]
        if key_cache != []:
            layer_keys = torch.cat(key_cache, dim=-2)
            layer_values = torch.cat(value_cache, dim=-2)
            new_cache.update(layer_keys, layer_values, idx)
    return new_cache


def test_crop_dynamic_cache():
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

def test_concat_dynamic_cache():
    cache1 = DynamicCache()
    cache2 = DynamicCache()
    x = torch.randn(1, 4, 10, 2)
    cache1.key_cache = [x for _ in range(10)]
    cache1.value_cache = [x for _ in range(10)]
    cache2.key_cache = [x for _ in range(10)]
    cache2.value_cache = [x for _ in range(10)]
    new_cache = concat_dynamic_cache([cache1, cache2])
    print(new_cache.key_cache[0].shape)
    print(torch.allclose(x, new_cache.key_cache[0][..., 0:10, :]))
    print(torch.allclose(x, new_cache.key_cache[0][..., 10:, :]))

if __name__ == "__main__":
    test_crop_dynamic_cache()
    print("-" * 100)
    test_concat_dynamic_cache()