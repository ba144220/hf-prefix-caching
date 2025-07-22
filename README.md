# HF Prefix Caching

Add a few lines to speed up your HuggingFace models with prefix caching.

```diff

+ prefix_cache = PrefixCache()

for batch in batches # inference loop ...
+   kv_cache = prefix_cache.get(input_ids)

    # Forward pass with cached key-values
    outputs = model.forward(
        ...
+       use_cache=True,
+       past_key_values=kv_cache,
    )

+   prefix_cache.update(input_ids, outputs.past_key_values)
```

### Features

- **Significant Speedup**: **2-3x speedup** when dealing datasets with repeated prefixes, such as `MMLU`
- **Easy Integration**: Add only a few lines of code to your existing inference loop
- **Automatic Cache Management**: Automatically manage the cache size and remove the least used prefixes
- **Chunked Caching**: Divide KV caches into smaller blocks, preventing from taking up big contiguous memory blocks

## Installation

```bash
git clone https://github.com/ba144220/hf-prefix-caching.git
cd hf-prefix-caching
pip install -e .
```

## Usage

### Basic Setup

```python
from transformers import AutoTokenizer
from hf_prefix_caching.prefix_cache import PrefixCache

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer.padding_side = "right"

# Create a prefix cache
prefix_cache = PrefixCache(
    block_size=64,                        # Size of each cache block
    padding_side=tokenizer.padding_side,  # Currently, we only support right padding
    max_block_length=40,                  # Maximum blocks per sequence
    max_num_blocks=128,                   # Total cache capacity
    max_new_blocks=4,                     # Max new blocks per update
)
...
```

Here are the two main functions you need to use:

1. `get`: Search and get the cached key-values for the input
2. `update`: Update the cache with the new key-values

```python
...
for batch in batched_dataset:
    # Prepare your input
    inputs = tokenizer(batch["text"], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Search and get the cached key-values for the input
    kv_cache = prefix_cache.get(input_ids)

    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,

        # Pass the cached key-values to the model
        use_cache=True,
        past_key_values=kv_cache,
    )

    # Update cache with new key-values
    prefix_cache.update(input_ids, outputs.past_key_values)
...
```

### Advanced Usage (Pre-building Cache with Frequent Prefixes)

Apart from building the cache on the fly, you can also pre-build the cache with frequent prefixes, so that you don't have to call `update` in your inference loop.

We provide a function `get_frequent_prefixes` to get the frequent prefixes from your dataset.

```python
from hf_prefix_caching.utils.get_frequent_prefixes import get_frequent_prefixes

frequent_prefixes: List[Tuple[str, int]] = get_frequent_prefixes(
    dataset["text"],                   # List of texts
    min_prefix_len=10,       # Minimum prefix length (i.e. the prefixes will be at least 10 characters long)
    max_prefix_len=50000,    # Maximum prefix length (i.e. the prefixes will be at most 50000 characters long)
    step_size=16,            # Step size for prefix search (i.e. the prefixes will be searched in chunks of 16 characters)
    threshold=1,             # Minimum frequency threshold (i.e. the prefixes will be kept if they appear at least 1 time)
    max_results=100,         # Best N prefixes will be returned
)
```

The returned `frequent_prefixes` is a list of tuples, where the first element is the prefix and the second element is the frequency.
We sort the prefixes by `frequency * length` in descending order.

Then you can pre-populate the cache with the frequent prefixes.

```python
prefix_cache = PrefixCache(
    block_size=64,
    padding_side=tokenizer.padding_side,
    max_block_length=40,
    max_num_blocks=128,
    max_new_blocks=40,    # You may want to set a larger value during pre-building (i.e. build cache more aggressively)
)

for prefix, frequency in frequent_prefixes:
    inputs = tokenizer([prefix], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    kv_cache = prefix_cache.get(input_ids)
    kv_length = kv_cache.get_seq_length()

    remaining_input_ids = input_ids[:, kv_length:]

    outputs = model.forward(
        input_ids=remaining_input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=True,
    )
    
    prefix_cache.update(input_ids, outputs.past_key_values)
```

Finally, you can use this pre-built cache in your inference loop.

```python
for batch in batched_dataset:
    inputs = tokenizer(batch["text"], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    kv_cache = prefix_cache.get(input_ids)

    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=True,
        past_key_values=kv_cache,
    )

    # No need to update the cache here
```

## Performance

We tested the performance of the prefix cache on `MMLU` dataset (see `tests/test_with_mmlu.py`). We randomly sampled 2048 examples from the dataset. For the pre-built cache, the time includes the time of building the cache.

| Test                                | Shuffled | Batch Size | Time (s) | Speedup |
|-------------------------------------|----------|------------|----------|---------|
| Baseline                            | ❌       | 4          | 130.16   | 1.00x   |
| Prefix Caching (w/o Pre-built Cache)| ❌       | 4          | 69.97    | 1.86x   |
| Prefix Caching (w/ Pre-built Cache) | ❌       | 4          | 64.52    | **2.02x**   |
| Baseline                            | ✅       | 1          | 145.46   | 1.00x   |
| Prefix Caching (w/o Pre-built Cache)| ✅       | 1          | 157.94   | 0.92x   |
| Prefix Caching (w/ Pre-built Cache) | ✅       | 1          | 122.68   | **1.29x**   |

> [!NOTE]
> If the dataset is shuffled, set the batch size to 1 to prevent wasted padding tokens.
