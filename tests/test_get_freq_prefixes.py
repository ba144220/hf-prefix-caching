from utils.prepare_mmlu_dataset import prepare_mmlu_dataset
from hf_prefix_caching.utils.get_frequent_prefixes import get_frequent_prefixes
import time
import re
def main():
    dataset = prepare_mmlu_dataset(num_shots=5, dataset_num_proc=64)
    texts = dataset["text"]
    print(len(texts))
    start_time = time.perf_counter_ns()
    results = get_frequent_prefixes(texts,
        max_results=200,
        min_prefix_len=0,
        max_prefix_len=50000,
        step_size=8,
        threshold=10,
    )
    end_time = time.perf_counter_ns()
    
    print(len(results))
    for idx, (prefix, freq) in enumerate(results):
        print("-" * 100)
        print(f"{idx}: {freq} {len(prefix)}")

    subjects = set()
    for idx, (prefix, freq) in enumerate(results):
        # Get the subject of the prefix
        subject = re.search(r"The following are multiple choice questions \(with answers\) about (.*)\.", prefix)
        if subject:
            subject = subject.group(1)
            subjects.add(subject)
    subjects = sorted(list(subjects))
    print(subjects)
    print(len(subjects))


    print(f"Time taken: {(end_time - start_time) / 1e6} ms")


if __name__ == "__main__":
    main()