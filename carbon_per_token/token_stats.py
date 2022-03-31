import argparse
import logging
import random
import json
import torch
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from promptsource.templates import DatasetTemplates

# from t-zero
from data_collator import DataCollatorForMultipleChoice
from template_list import template_list

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bigscience/T0_3B",
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    token_counts = defaultdict(int)
    padding = "max_length" if args.pad_to_max_length else False
    isGpu = torch.cuda.is_available()
    device = "cuda:0" if isGpu else "cpu"

    for (dataset_name, dataset_config_name), template_names in template_list.items():
        
        # download dataset
        # NOTE: anli doesn't work, and story_cloze requires manual download, skipping for now
        if dataset_name == "anli" or dataset_name == "story_cloze":
            # raw_datasets = load_dataset(dataset_name, split=dataset_config_name)
            continue
        else:
            print(f"Download for {dataset_name} started at: {datetime.now()}")
            raw_datasets = load_dataset(dataset_name, dataset_config_name, split="validation")
        column_names = raw_datasets.column_names
        full_dataset_id = (
            f"{dataset_name}"
            if dataset_config_name is None
            else f"{dataset_name}/{dataset_config_name}"
        )
        prompts = DatasetTemplates(full_dataset_id)
        print(f"{dataset_name} finished downloading & loaded: {datetime.now()}")

        # get tokenizer
        print(f"Grabbing tokenizer for {args.tokenizer_name} at : {datetime.now()}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        print(f" Tokenizer downloaded, grabbing pretrained model for {args.tokenizer_name} at: {datetime.now()}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.tokenizer_name)
        model = model.to(device)
        
        if tokenizer.pad_token is None:
            for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
                if token is not None:
                    tokenizer.pad_token = token
            if tokenizer.pad_token is None:
                raise ValueError("Please define a pad token id.")
        
        for template_name in template_names:
            
            template = prompts[template_name]

            def preprocess_inference(examples):
                """ A helper function used to """
                bs = len(examples[column_names[0]])

                input_texts = []
                target_texts = []
                answer_choices_texts = []
                for i in range(bs):
                    ex = {
                        k: examples[k][i]
                        for k in column_names
                    }
                    input, target = template.apply(ex)
                    ex_answer_choices = template.get_answer_choices_list(ex)
                    assert target in ex_answer_choices
                    input_texts.append(input)
                    target_texts.append(target)
                    answer_choices_texts.append(ex_answer_choices)

                tokenized_inputs = tokenizer(
                    input_texts,
                    padding=padding,
                    max_length=args.max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                tokenized_targets = [
                    tokenizer(
                        ans_choi,
                        padding=True,
                        max_length=args.target_max_length,
                        truncation=True,
                    )
                    for ans_choi in answer_choices_texts
                ]

                return [tokenized_inputs, tokenized_targets]

            def preprocess_function(examples):
                bs = len(examples[column_names[0]])

                input_texts = []
                target_texts = []
                answer_choices_texts = []
                for i in range(bs):
                    ex = {
                        k: examples[k][i]
                        for k in column_names
                    }
                    input, target = template.apply(ex)
                    ex_answer_choices = template.get_answer_choices_list(ex)
                    assert target in ex_answer_choices
                    input_texts.append(input)
                    target_texts.append(target)
                    answer_choices_texts.append(ex_answer_choices)

                tokenized_inputs = tokenizer(
                    input_texts,
                    padding=padding,
                    max_length=args.max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                tokenized_targets = [
                    tokenizer(
                        ans_choi,
                        padding=True,
                        max_length=args.target_max_length,
                        truncation=True,
                    )
                    for ans_choi in answer_choices_texts
                ]

                features = {
                    k: [
                        [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                        for idx, elem in enumerate(v)
                    ]
                    for k, v in tokenized_inputs.items()
                }

                features["labels"] = [
                    tokenized_targets[idx]["input_ids"]
                    for idx in range(bs)
                ]
                features["labels_attention_mask"] = [
                    tokenized_targets[idx]["attention_mask"]
                    for idx in range(bs)
                ]
                features["targets"] = [
                    answer_choices_texts[idx].index(t)
                    for idx, t in enumerate(target_texts)
                ]

                return features

            try:
                eval_dataset = raw_datasets.map(
                    preprocess_function, batched=True, remove_columns=column_names
                )
            except:
                eval_dataset = raw_datasets.map(preprocess_function, batched=True)

            # Log a few random samples from the eval set:
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

            # DataLoaders creation:
            if args.pad_to_max_length:
                # If padding was already done to max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorForMultipleChoice(
                    tokenizer, pad_to_multiple_of=None
                )

            eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

            for line in eval_dataloader:
                token_counts[full_dataset_id] += len(line["input_ids"])
    
    with open("token_counts.json", "w") as f:
        f.write(json.dumps(dict(token_counts)))

if __name__ == "__main__":
    main()