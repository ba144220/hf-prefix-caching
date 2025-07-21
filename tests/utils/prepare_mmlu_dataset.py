from datasets import Dataset
from typing import Dict, List
from datasets import load_dataset
import pandas as pd
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class MMLUTask:
    text_col_name: str = "text"
    task_name: str = "mmlu"
    def __init__(self, num_shots: int = 0, num_proc: int = 1):
        self.num_shots = num_shots
        self.num_proc = num_proc

        # prepare dataset
        dataset_dict = load_dataset("cais/mmlu", "all")
        dataset: Dataset = dataset_dict["test"] # type: ignore
        dev_dataset: Dataset = dataset_dict["dev"] # type: ignore

        
        # prepare prefixes
        self.prefixes = self._prepare_prefixes(dev_dataset)

        # prepare dataset
        self.dataset = self._prepare_dataset(dataset)

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_prefixes(self) -> Dict[str, str]:
        return self.prefixes

    def get_answer_map(self) -> Dict[str, int]:
        return {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            " A": 0,
            " B": 1,
            " C": 2,
            " D": 3,
        }


    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        prefixes = self.prefixes
        text_col_name = self.text_col_name

        def preprocess_fn(example, prefixes: Dict[str, str], text_col_name: str):
            subject = example["subject"]
            question = example["question"]
            choices = example["choices"]
            answer = ["A", "B", "C", "D"][example["answer"]]
            text = f"{prefixes[subject]}\n\n{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            return {
                text_col_name: text
            }
        
        dataset = dataset.map(
            preprocess_fn, 
            batched=False, 
            num_proc=self.num_proc, 
            fn_kwargs={
                "prefixes": prefixes, 
                "text_col_name": text_col_name
            }
        )
        return dataset

    def _prepare_prefixes(self, dev_dataset: Dataset) -> Dict[str, str]:

        dev_df: pd.DataFrame = dev_dataset.to_pandas()
        subjects = dev_df["subject"].unique()
        prefixes = {}

        for subject in subjects:
            subject_examples = dev_df[dev_df["subject"] == subject]

            sampled_examples = subject_examples.head(min(self.num_shots, len(subject_examples)))
            prefix_lines = [
                f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}."
            ]
            for _, example in sampled_examples.iterrows():
                question = example["question"]
                choices = example["choices"]
                answer = ["A", "B", "C", "D"][example["answer"]]
                prefix_lines.append(f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}")
            prefixes[subject] = "\n\n".join(prefix_lines)

        return prefixes


def prepare_mmlu_dataset(tokenizer: PreTrainedTokenizerBase, num_shots: int = 0, optimize_batching: bool = False, dataset_num_proc: int = 1) -> Dataset:
    mmlu_task = MMLUTask(num_shots=num_shots, num_proc=dataset_num_proc)
    dataset = mmlu_task.get_dataset()

    if optimize_batching:
        def get_tokenized_length(example):
            length = len(tokenizer.encode(example["text"]))
            return {
                "length": length
            }
        
        dataset = dataset.map(get_tokenized_length, batched=False, num_proc=dataset_num_proc)
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns(["length"])
    return dataset