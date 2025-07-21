from utils.prepare_mmlu_dataset import prepare_mmlu_dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from datasets import Dataset
from contextlib import contextmanager
import time
from tqdm import tqdm
from hf_prefix_caching.prefix_cache import PrefixCache

MODEL_NAME = "Qwen/Qwen3-8B"

@contextmanager
def timeit(name: str):
    start_time = time.perf_counter_ns()
    yield
    end_time = time.perf_counter_ns()
    print(f"{name} took {(end_time - start_time) / 1e6} ms")

def baseline_inference(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    batched_dataset: Dataset,
) -> None:
    for batch in tqdm(batched_dataset, desc="Baseline Inference"):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            logits_to_keep=1,
        )

def prefix_caching_inference(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    batched_dataset: Dataset,
    prefix_cache: PrefixCache,
) -> None:
    
    for batch in tqdm(batched_dataset, desc="Prefix Caching Inference"):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        kv_cache = prefix_cache.get(input_ids)

        kv_length = kv_cache.get_seq_length()
        remaining_input_ids = input_ids[:, kv_length:]

        # print(f"Hit sequence length: {kv_length}")

        outputs: CausalLMOutputWithPast = model.forward(
            input_ids=remaining_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            logits_to_keep=1,
            use_cache=True,
            past_key_values=kv_cache,
        )

        prefix_cache.update(input_ids, outputs.past_key_values)


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"

    model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    dataset = prepare_mmlu_dataset(tokenizer, optimize_batching=True, dataset_num_proc=64, num_shots=5)

    dataset = dataset.select(range(256))
    batched_dataset = dataset.batch(batch_size=8) 

    with timeit("baseline_inference"):
        baseline_inference(model, tokenizer, batched_dataset)

    prefix_cache = PrefixCache(
        block_size=64,
        padding_side=tokenizer.padding_side,
        max_block_length=50,
        max_num_blocks=128,
    )

    with timeit("prefix_caching_inference"):
        prefix_caching_inference(model, tokenizer, batched_dataset, prefix_cache)

    # for cache_hash, cache_block in prefix_cache.caches.items():
    #     print(cache_hash[:10], cache_block)    



if __name__ == "__main__":
    main()