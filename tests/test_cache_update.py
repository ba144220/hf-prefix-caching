import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
from hf_prefix_caching.prefix_cache import PrefixCache
from transformers.cache_utils import DynamicCache

MODEL_NAME = "Qwen/Qwen3-8B"
@torch.no_grad()
def main():
    model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    inputs = tokenizer(["Hello, world!", "Hello, I'm a cat.", "Hello, I'm a dog."], padding=True, truncation=True, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    print(input_ids.shape)

    cache = DynamicCache()
    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    prefix_cache = PrefixCache(block_size=2, padding_side=tokenizer.padding_side)
    prefix_cache.update(input_ids, output.past_key_values)
    for key, cache_block in prefix_cache.caches.items():
        print(key[:10], cache_block)

    
    

if __name__ == "__main__":
    main()