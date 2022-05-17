import argparse

import torch
from tqdm.auto import tqdm
from codecarbon import EmissionsTracker
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, default_data_collator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-v1")
    parser.add_argument("--num_trials", type=int, default=10, help="how many times to repeat the experiment")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=512, help="sequence length")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--from_pretrained", action="store_true", help="whether to load pretrained model")
    args = parser.parse_args()
    return args


def prepare_dataset(tokenizer, dataset_name, dataset_config, block_size, batched=True, num_proc=2):
    raw_datasets = load_dataset(dataset_name, dataset_config)
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenize_function = lambda examples, tokenizer: tokenizer(examples[text_column_name])
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=batched,
        num_proc=num_proc,
        remove_columns=column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=batched,
        num_proc=num_proc,
        fn_kwargs={"block_size": block_size},
    )
    return lm_datasets


def group_texts(examples, block_size):
    # concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # drop the small remainder
    total_length = (total_length // block_size) * block_size
    # split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


# NOTE: compare with torch.no_grad
@torch.inference_mode()
def main(args):
    device = torch.device(args.device)
    if args.from_pretrained:
        model = AutoModel.from_pretrained(args.model_name).to(device)
    else:
        # randomly initialize model
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModel.from_config(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = prepare_dataset(tokenizer, args.dataset_name, args.dataset_config, block_size=args.block_size)
    # TODO: check if `shuffle` should be `True`
    dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    pretrained_flag = "pretrained" if args.pretrained else "random"
    project_name = f"{args.model_name}_{pretrained_flag}_{args.dataset_name}_{args.batch_size}"
    tracker = EmissionsTracker(project_name=project_name, log_level="error")
    for _ in range(args.num_trials):
        for batch in tqdm(dataloader):
            inputs = {key: value.to(model.device) for key, value in batch.items()}
            tracker.start()
            _ = model(**inputs)
            tracker.stop()


if __name__ == "__main__":
    args = get_args()
    main(args)