import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# 🚀 Define a single GENERATION_CONFIGS dictionary with both decoding methods
GENERATION_CONFIGS = {
    # ✅ Top-p Sampling (original setting)
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,  # 🚨 Beam Search is OFF here
    },

    # ✅ Beam Search (New Configuration)
    "beam_search": {
        "max_new_tokens": 400,  # 🔹 Increase max tokens for better summaries
        "do_sample": False,  # 🔹 Disable sampling for deterministic output
        "num_beams": 8,  # 🔹 Enable Beam Search (5 beams)
        "early_stopping": False,
        "length_penalty": 1.2,
    },

    # ✅ Other Sampling Variations (unchanged)
    **{
        f"sampling_topp_{str(topp).replace('.', '')}": {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_return_sequences": 8,
            "top_p": topp,
        }
        for topp in [0.5, 0.8, 0.95, 0.99]
    },
}


# add base.csv config to all configs
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        # "max_length": 2048,
        "min_length": 0,
        "early_stopping": True,
        **value,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--decoding_config", type=str, default="beam_search", choices=GENERATION_CONFIGS.keys())
    #parser.add_argument("--decoding_config", type=str, default="top_p_sampling", choices=GENERATION_CONFIGS.keys())

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trimming", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--output_dir", type=str, default="data/candidates")

    # if ran in a scripted way, the output path will be printed
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    
    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args


def prepare_dataset(dataset_path) -> Dataset:
    try:
        dataset = pd.read_csv(dataset_path)
    except:
        raise ValueError(f"Unknown dataset {dataset_path}")

    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer(
    model, tokenizer, dataset: Dataset, decoding_config, batch_size: int,
    device: str, trimming: bool
) -> Dataset:
    """
    @param model: The model used to generate the summaries
    @param tokenizer: The tokenizer used to tokenize the text and the summary
    @param dataset: A dataset with the text
    @param decoding_config: Dictionary with the decoding config
    @param batch_size: The batch size used to generate the summaries
    @return: The same dataset with the summaries added
    """
    # create a dataset with the text and the summary

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)

    # generate summaries
    summaries = []
    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = batch["text"]

        inputs = tokenizer(
            text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # generate summaries
        outputs = model.generate(
            **inputs,
            **decoding_config,
        )
        
        total_size = outputs.numel()  # Total number of elements in the tensor
        target_size = batch_size * outputs.shape[-1]  # Target size of the last dimension
        pad_size = (target_size - (total_size % target_size)) % target_size  # Calculate the required padding size to make the total number of elements divisible by the target size

        # Pad the tensor with zeros to make the total number of elements divisible by the target size
        if not trimming and pad_size != 0: outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_size // outputs.shape[-1]))

        # output : (batch_size * num_return_sequences, max_length)
        try:
            outputs = outputs.reshape(batch_size, -1, outputs.shape[-1])
        except Exception as e:
            print(f"Error reshaping outputs: {e}")
            raise ValueError(f"Cannot reshape tensor of size {outputs.numel()} into shape "
                            f"({batch_size}, -1, {outputs.shape[-1]}).")
        
        # decode summaries
        for b in range(batch_size):
            summaries.append(
                [
                    tokenizer.decode(
                        outputs[b, i],
                        skip_special_tokens=True,
                    )
                    for i in range(outputs.shape[1])
                ]
            )

    # if trimming the last batch, remove them from the dataset
    if trimming: dataset = dataset.select(range(len(summaries)))
    
    # add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    
    return dataset


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


def main():
    args = parse_args()

    # load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # move model to device
    model = model.to(args.device)

    # load the dataset
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries
    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
        args.device,
        args.trimming,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode('summary')
    df_dataset = df_dataset.reset_index()
    # add an idx with  the id of the summary for each example
    df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    padding_status = "trimmed" if args.trimming else "padded"
    output_path = (
        Path(args.output_dir)
        / f"{model_name}-_-{args.dataset_path.stem}-_-{args.decoding_config}-_-{padding_status}-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")

    # in case of scripted run, print the output path
    if args.scripted_run: print(output_path)


if __name__ == "__main__":
    main()
 


'''
import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# 🚀 Define both decoding methods
GENERATION_CONFIGS = {
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,  # 🚨 Beam Search is OFF
    },

    "beam_search": {
        "max_new_tokens": 400,  # 🚀 Allow longer summaries
        "do_sample": False,  # 🚀 Turns OFF sampling for deterministic output
        "num_beams": 4,  # 🚀 Increase beam width for better candidate selection
        "early_stopping": True,  # ✅ Ensure full-length summaries
        "length_penalty": 1.0,  # ✅ Prevent excessive shortening
    },
}

# ✅ Ensure early_stopping=False is preserved
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        "min_length": 0,
        "early_stopping": value.get("early_stopping", True),  # ✅ Keeps intended settings
        **value,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--decoding_config", type=str, default="beam_search", choices=GENERATION_CONFIGS.keys())

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trimming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="data/candidates")
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()

def prepare_dataset(dataset_path) -> Dataset:
    dataset = pd.read_csv(dataset_path)
    return Dataset.from_pandas(dataset)

def evaluate_summarizer(model, tokenizer, dataset: Dataset, decoding_config, batch_size: int, device: str, trimming: bool) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)
    summaries = []

    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = batch["text"]
        inputs = tokenizer(
            text,
            max_length=1024,  
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, **decoding_config)

        # Ensure outputs shape is correct before indexing
        num_returned_sequences = outputs.shape[0]  # Get actual number of generated sequences

        for b in range(batch_size):
            generated_summaries = []
            for i in range(min(num_returned_sequences, decoding_config.get("num_return_sequences", 1))):  # Ensure valid index
                generated_summaries.append(
                    tokenizer.decode(outputs[i], skip_special_tokens=True)
                )

            summaries.append(generated_summaries)

    if trimming:
        dataset = dataset.select(range(len(summaries)))

    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    return dataset

'''
'''
#original def
def evaluate_summarizer(model, tokenizer, dataset: Dataset, decoding_config, batch_size: int, device: str, trimming: bool) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)
    summaries = []

    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = batch["text"]
        inputs = tokenizer(
            text,
            max_length=1024,  
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, **decoding_config)

        for b in range(batch_size):
            summaries.append([
                tokenizer.decode(outputs[b, i], skip_special_tokens=True)
                for i in range(outputs.shape[1])
            ])

    if trimming:
        dataset = dataset.select(range(len(summaries)))

    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    return dataset
'''

'''
def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")
'''
'''
def fix_ending(summary):
    if isinstance(summary, str) and summary[-1] not in {".", "?", "!"}:  
        return summary + "."  # ✅ Ensure proper sentence endings
    return summary
'''
'''
def fix_ending(summary):
  if isinstance(summary, str) and summary and summary[-1] not in {".", "?", "!"}:
    return summary + "."
  return summary

def main():
    args = parse_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.half()  # Enable FP16 (Half-Precision)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model = model.to(args.device)

    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
        args.device,
        args.trimming,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode('summary').reset_index()
    df_dataset["id_candidate"] = df_dataset.groupby(["index"]).cumcount()

    # ✅ Fix sentence endings
    df_dataset["summary"] = df_dataset["summary"].apply(fix_ending)

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    output_path = Path(args.output_dir) / f"{model_name}-_-{args.dataset_path.stem}-_-{args.decoding_config}-_-trimmed-_-{date}.csv"

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")
    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()

'''

'''
import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# 🚀 Define a single GENERATION_CONFIGS dictionary with both decoding methods
GENERATION_CONFIGS = {
    # ✅ Top-p Sampling (original setting)
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,  # 🚨 Beam Search is OFF here
    },

    # ✅ Beam Search (New Configuration)
    "beam_search": {
        "max_new_tokens": 400,  # 🔹 Increase max tokens for better summaries
        "do_sample": False,  # 🔹 Disable sampling for deterministic output
        "num_beams": 8,  # 🔹 Enable Beam Search (5 beams)
        "early_stopping": False,
        "length_penalty": 1.2,
    },

    # ✅ Other Sampling Variations (unchanged)
    **{
        f"sampling_topp_{str(topp).replace('.', '')}": {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_return_sequences": 8,
            "top_p": topp,
        }
        for topp in [0.5, 0.8, 0.95, 0.99]
    },
}


# add base.csv config to all configs
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        # "max_length": 2048,
        "min_length": 0,
        "early_stopping": True,
        **value,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--decoding_config", type=str, default="beam_search", choices=GENERATION_CONFIGS.keys())
    #parser.add_argument("--decoding_config", type=str, default="top_p_sampling", choices=GENERATION_CONFIGS.keys())

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trimming", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--output_dir", type=str, default="data/candidates")

    # if ran in a scripted way, the output path will be printed
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    
    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args


def prepare_dataset(dataset_path) -> Dataset:
    try:
        dataset = pd.read_csv(dataset_path)
    except:
        raise ValueError(f"Unknown dataset {dataset_path}")

    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer(
    model, tokenizer, dataset: Dataset, decoding_config, batch_size: int,
    device: str, trimming: bool
) -> Dataset:
    """
    @param model: The model used to generate the summaries
    @param tokenizer: The tokenizer used to tokenize the text and the summary
    @param dataset: A dataset with the text
    @param decoding_config: Dictionary with the decoding config
    @param batch_size: The batch size used to generate the summaries
    @return: The same dataset with the summaries added
    """
    # create a dataset with the text and the summary

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)

    # generate summaries
    summaries = []
    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = batch["text"]

        inputs = tokenizer(
            text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # generate summaries
        outputs = model.generate(
            **inputs,
            **decoding_config,
        )
        
        total_size = outputs.numel()  # Total number of elements in the tensor
        target_size = batch_size * outputs.shape[-1]  # Target size of the last dimension
        pad_size = (target_size - (total_size % target_size)) % target_size  # Calculate the required padding size to make the total number of elements divisible by the target size

        # Pad the tensor with zeros to make the total number of elements divisible by the target size
        if not trimming and pad_size != 0: outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_size // outputs.shape[-1]))

        # output : (batch_size * num_return_sequences, max_length)
        try:
            outputs = outputs.reshape(batch_size, -1, outputs.shape[-1])
        except Exception as e:
            print(f"Error reshaping outputs: {e}")
            raise ValueError(f"Cannot reshape tensor of size {outputs.numel()} into shape "
                            f"({batch_size}, -1, {outputs.shape[-1]}).")
        
        # decode summaries
        for b in range(batch_size):
            summaries.append(
                [
                    tokenizer.decode(
                        outputs[b, i],
                        skip_special_tokens=True,
                    )
                    for i in range(outputs.shape[1])
                ]
            )

    # if trimming the last batch, remove them from the dataset
    if trimming: dataset = dataset.select(range(len(summaries)))
    
    # add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    
    return dataset


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


def main():
    args = parse_args()

    # load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    # move model to device
    model = model.to(args.device)

    # load the dataset
    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries
    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
        args.device,
        args.trimming,
    )

    df_dataset = dataset.to_pandas()
    df_dataset = df_dataset.explode('summary')
    df_dataset = df_dataset.reset_index()
    # add an idx with  the id of the summary for each example
    df_dataset['id_candidate'] = df_dataset.groupby(['index']).cumcount()

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    padding_status = "trimmed" if args.trimming else "padded"
    output_path = (
        Path(args.output_dir)
        / f"{model_name}-_-{args.dataset_path.stem}-_-{args.decoding_config}-_-{padding_status}-_-{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_csv(output_path, index=False, encoding="utf-8")

    # in case of scripted run, print the output path
    if args.scripted_run: print(output_path)


if __name__ == "__main__":
    main()
    '''