import random
from utils.prepare_mmlu_dataset import prepare_mmlu_dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from datasets import Dataset
from contextlib import contextmanager
import time
from tqdm import tqdm
from hf_prefix_caching.prefix_cache import PrefixCache
from hf_prefix_caching.utils.get_frequent_prefixes import get_frequent_prefixes

MODEL_NAME = "Qwen/Qwen3-8B"

@contextmanager
def timeit(name: str):
    start_time = time.perf_counter_ns()
    yield
    end_time = time.perf_counter_ns()
    print(f"{name} took {(end_time - start_time) / 1e9} s")

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
    update: bool = True,
) -> None:
    get_time = 0
    update_time = 0
    for i, batch in enumerate(tqdm(batched_dataset, desc="Prefix Caching Inference")):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        start_time = time.perf_counter_ns()
        kv_cache = prefix_cache.get(input_ids)
        get_time += time.perf_counter_ns() - start_time

        kv_length = kv_cache.get_seq_length()
        remaining_input_ids = input_ids[:, kv_length:]
        # print(f"hit length: {kv_length}/{input_ids.shape[1]}")

        outputs: CausalLMOutputWithPast = model.forward(
            input_ids=remaining_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            logits_to_keep=1,
            use_cache=True,
            past_key_values=kv_cache,
        )
        if update:
            start_time = time.perf_counter_ns()
            prefix_cache.update(input_ids, outputs.past_key_values)
            update_time += time.perf_counter_ns() - start_time

    print(f"Get time: {get_time / 1e6} ms")
    print(f"Update time: {update_time / 1e6} ms")


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"

    model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    dataset = prepare_mmlu_dataset(tokenizer=tokenizer, optimize_batching=True, dataset_num_proc=64, num_shots=5)

    # dataset = dataset.select(range(1024))
    # Randomly sample 2048 examples, but in order
    dataset = dataset.select(sorted(random.sample(range(len(dataset)), 2048)))
    batched_dataset = dataset.batch(batch_size=8) 

    with timeit("baseline_inference"):
        baseline_inference(model, tokenizer, batched_dataset)

    prefix_cache = PrefixCache(
        block_size=64,
        padding_side=tokenizer.padding_side,
        max_block_length=40,
        max_num_blocks=128,
        max_new_blocks=4,
    )

    with timeit("prefix_caching_inference"):
        prefix_caching_inference(model, tokenizer, batched_dataset, prefix_cache, update=True)

    prefix_cache = PrefixCache(
        block_size=64,
        padding_side=tokenizer.padding_side,
        max_block_length=40,
        max_num_blocks=512,
        max_new_blocks=40,
    )

    with timeit("prefix_caching_inference"):

        # Get the frequent prefixes
        frequent_prefixes = get_frequent_prefixes(
            dataset["text"],
            min_prefix_len=10,
            max_prefix_len=50000,
            step_size=16,
            threshold=1,
            max_results=100,
        )
        # Reverse the frequent prefixes
        frequent_prefixes = frequent_prefixes[::-1]

        for prefix, _ in tqdm(frequent_prefixes, desc="Building prefix cache"):
            inputs = tokenizer([prefix], padding=True, truncation=True, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"].long()
            attention_mask = inputs["attention_mask"].long()
            kv_cache = prefix_cache.get(input_ids)
            outputs: CausalLMOutputWithPast = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                logits_to_keep=1,
                use_cache=True,
                past_key_values=kv_cache,
            )
            prefix_cache.update(input_ids, outputs.past_key_values)

        prefix_caching_inference(model, tokenizer, batched_dataset, prefix_cache, update=False)
    
    


if __name__ == "__main__":
    main()