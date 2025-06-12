import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 🚀 Define T5 Decoding Configurations
GENERATION_CONFIGS = {
    "beam_search": {
        "max_new_tokens": 512,  # ✅ T5 supports slightly longer summaries
        "do_sample": False,
        "num_beams": 5,
        "early_stopping": True,
        "length_penalty": 1.2,
    },
    "top_p_sampling": {
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,
    },
}

# ✅ Ensure early_stopping=False is preserved
for key, value in GENERATION_CONFIGS.items():
    GENERATION_CONFIGS[key] = {
        "min_length": 50,  # ✅ Ensure minimum summary length
        "early_stopping": value.get("early_stopping", True),
        **value,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5-large")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2020.csv")
    parser.add_argument("--decoding_config", type=str, default="beam_search", choices=GENERATION_CONFIGS.keys())

    parser.add_argument("--batch_size", type=int, default=16)  # ✅ T5 is more memory-efficient
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trimming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="data/t5_candidates")
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()

def prepare_dataset(dataset_path) -> Dataset:
    dataset = pd.read_csv(dataset_path)

    if "text" not in dataset.columns:
        raise ValueError(f"Dataset must contain a 'text' column. Found: {dataset.columns}")

    return Dataset.from_pandas(dataset)

def evaluate_summarizer(model, tokenizer, dataset: Dataset, decoding_config, batch_size: int, device: str, trimming: bool) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=trimming)
    summaries = []

    print("Generating summaries...")

    for batch in tqdm(dataloader):
        text = ["summarize: " + t for t in batch["text"]]  # ✅ Add T5 Prefix

        inputs = tokenizer(
            text,
            max_length=1024,  # ✅ Keep max input length
            padding="max_length",  # ✅ Fixed padding for T5
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            **decoding_config,
        )

        # ✅ Ensure all batches are properly processed
        for b in range(outputs.shape[0]):
            summaries.append(tokenizer.decode(outputs[b], skip_special_tokens=True))

    if trimming:
        dataset = dataset.select(range(len(summaries)))

    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})
    return dataset

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

def fix_ending(summary):
    if isinstance(summary, str) and summary[-1] not in {".", "?", "!"}:  
        return summary + "."  
    return summary

def main():
    args = parse_args()

    # ✅ Load T5 model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ✅ T5 fix: Ensure PAD token exists
    tokenizer.pad_token = tokenizer.eos_token  # ✅ T5 needs EOS as PAD

    model = model.to(args.device)

    print(f"🔹 Running summarization with {args.model_name} (max_length=1024)")

    # ✅ Load dataset
    dataset = prepare_dataset(args.dataset_path)

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # ✅ Generate summaries
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

    # ✅ Save results
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)

    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}-_-{args.dataset_path.stem}-_-{args.decoding_config}-_-trimmed-_-{date}.csv"
    
    print(f"✅ Saving output to: {output_path}")  
    df_dataset.to_csv(output_path, index=False, encoding="utf-8")

    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()
