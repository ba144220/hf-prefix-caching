import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
from hf_prefix_caching.prefix_cache import PrefixCache
from transformers.cache_utils import DynamicCache
from utils.prepare_mmlu_dataset import prepare_mmlu_dataset
from datasets import Dataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_NAME = "Qwen/Qwen3-8B"


def baseline_inference(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    batched_dataset: Dataset,
) -> None:
    all_logits = []
    for batch in tqdm(batched_dataset, desc="Baseline Inference"):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_pos = torch.sum(attention_mask, dim=1) - 1
        logits = outputs.logits[torch.arange(len(last_pos)), last_pos]
        all_logits.append(logits)
    # Convert all_logits to a tensor
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits

def prefix_caching_inference(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    batched_dataset: Dataset,
    prefix_cache: PrefixCache,
    update: bool = True,
) -> None:

    all_logits = []

    for _, batch in enumerate(tqdm(batched_dataset, desc="Prefix Caching Inference")):
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        kv_cache = prefix_cache.get(input_ids)

        kv_length = kv_cache.get_seq_length()
        remaining_input_ids = input_ids[:, kv_length:]


        outputs: CausalLMOutputWithPast = model.forward(
            input_ids=remaining_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
            past_key_values=kv_cache,
        )
        if update:
            prefix_cache.update(input_ids, outputs.past_key_values)
        last_pos = torch.sum(attention_mask, dim=1) - 1 - kv_length
        logits = outputs.logits[torch.arange(len(last_pos)), last_pos]
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits



@torch.no_grad()
def main():
    dataset = prepare_mmlu_dataset(num_shots=5, dataset_num_proc=64, optimize_batching=True)
    dataset = dataset.select(range(128))
    batched_dataset = dataset.batch(batch_size=8)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    tokenizer.padding_side = "right"

    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.float32, device_map="cuda:0")
    model.eval()

    baseline_logits = baseline_inference(model, tokenizer, batched_dataset)
    print(baseline_logits.shape)
    prefix_cache = PrefixCache(
        block_size=64,
        padding_side=tokenizer.padding_side,
        max_block_length=40,
        max_num_blocks=512,
        max_new_blocks=40,
    )
    prefix_cache_logits = prefix_caching_inference(model, tokenizer, batched_dataset, prefix_cache, update=True)
    print(prefix_cache_logits.shape)

    # Check if the logits are the same
    diff = torch.abs(baseline_logits - prefix_cache_logits)
    print(f"logit max: {baseline_logits.max()}")
    print(f"logit min: {baseline_logits.min()}")
    print(f"logit mean: {baseline_logits.mean()}")
    print(f"logit std: {baseline_logits.std()}")
    print(f"diff max: {diff.max()}")
    print(f"diff mean: {diff.mean()}")
    print(f"diff std: {diff.std()}")


if __name__ == "__main__":
    main()