import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse
from tqdm import tqdm

from transformers import BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default="")

    args = parser.parse_args()
    return args



import pandas as pd
from pathlib import Path

def prepare_dataset(dataset):
    dataset_path = Path(dataset)

    # ✅ If the argument is a file path, load it directly
    if dataset_path.exists() and dataset_path.is_file():
        print(f"📂 Loading dataset from file: {dataset_path}")  # Debugging
        return pd.read_csv(dataset_path)

    # ✅ If the argument is a dataset name, use predefined paths
    dataset_dir = Path("rsasumm/data/processed/")
    dataset_files = {
        "amazon": "amazon_test.csv",
        "space": "space.csv",
        "yelp": "yelp_test.csv",
        "reviews": "test_metareviews.csv",
    }

    if dataset not in dataset_files:
        raise ValueError(f"❌ Unknown dataset '{dataset}' or file does not exist.")

    return pd.read_csv(dataset_dir / dataset_files[dataset])



'''
def prepare_dataset(dataset_name, dataset_path="rsasumm/data/processed/"):
    dataset_path = Path(dataset_path)
    if dataset_name == "amazon":
        dataset = pd.read_csv(dataset_path / "amazon_test.csv")
    elif dataset_name == "space":
        dataset = pd.read_csv(dataset_path / "space.csv")
    elif dataset_name == "yelp":
        dataset = pd.read_csv(dataset_path / "yelp_test.csv")
    elif dataset_name == "reviews":
        dataset = pd.read_csv(dataset_path / "test_metareviews.csv")
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


    return dataset
'''

# group text by sample id and concatenate text

def group_text_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the text by the sample id and concatenate the text.
    :param df: The dataframe
    :return: The dataframe with the text grouped by the sample id
    """
    texts = df.groupby("id")["text"].apply(lambda x: " ".join(x))

    # retrieve first gold by id
    gold = df.groupby("id")["gold"].first()

    # create new dataframe
    df = pd.DataFrame({"text": texts, "gold": gold}, index=texts.index)

    return df

def generate_summaries(model, tokenizer, df, batch_size, device):

    def make_instruction(text):
        return f"[INST]\n{text}\n Summarize the previous text:[/INST]\n\n"

    df["instruction"] = df["text"].apply(make_instruction)

    dataset = df["instruction"].tolist()  # ✅ Convert to a simple list of strings

    #model = model.to(device).eval()
    model = model.eval()
    summaries = []

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Summaries"):
            batch = dataset[i : i + batch_size]  # ✅ Correct batching
            
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)  # ✅ FIXED

            outputs = model.generate(
                **inputs, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=500
            )

            decoded_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(decoded_summaries)

    df["summary"] = [re.sub(r"\[INST\]\n.*\[/INST\]\n\n", "", summary) for summary in summaries]
    
    return df


def main():
    args = parse_args()
    model_name = "togethercomputer/Llama-2-7B-32K-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # ✅ Enable 8-bit model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, quantization_config=quantization_config, device_map="auto"
    )

    df = prepare_dataset(args.dataset)
    df = group_text_by_id(df)
    df = generate_summaries(model, tokenizer, df, args.batch_size, args.device)

    df['metadata/Method'] = "LLM"
    df['metadata/Model'] = model_name  # ✅ Fix: Use model name, not model object

    name = f"{args.dataset}-_-{model_name.replace('/', '-')}-_-llm_summaries.csv"
    path = Path(args.output) / name

    Path(args.output).mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)  # ✅ Fix: Remove unnecessary index

    print(f"✅ Summaries saved to {path}")

# ✅ Fix: Ensure script runs only when executed directly
if __name__ == "__main__":
    main()



'''
def generate_summaries(model, tokenizer, df, batch_size, device):

    # df columns = id, text, gold
    # make instruction:

    def make_instruction(text):
        return f"[INST]\n{text}\n Summarize the previous text:[/INST]\n\n"

    df["instruction"] = df["text"].apply(make_instruction)

    # make data loader
    dataset = df[["instruction"]].values.tolist()


    model = model.to(device).eval()

    summaries = []
    with torch.no_grad():
        for batch in tqdm(dataset):
            print(batch)
            inputs = tokenizer.encode(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=500)
            summaries.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # remove the instruction from the summaries
    df["summary"] = [re.sub(r"\[INST\]\n.*\[/INST\]\n\n", "", summary) for summary in summaries]

    return df
'''
'''
model_name = "togethercomputer/Llama-2-7B-32K-Instruct"  # ✅ Store the model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)  
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, quantization_config=quantization_config, device_map="auto"
)

# ✅ Use model_name instead of model object
name = f"{args.dataset}-_-{model_name.replace('/', '-')}-_-llm_summaries.csv"


'''
'''
def main():

    args = parse_args()
    model_name = "togethercomputer/Llama-2-7B-32K-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # ✅ Enable 8-bit model
    model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, quantization_config=quantization_config, device_map="auto"
                 )
    df = prepare_dataset(args.dataset)

    df = group_text_by_id(df)

    df = generate_summaries(model, tokenizer, df, args.batch_size, args.device)
    df['metadata/Method'] = "LLM"
    df['metadata/Model'] = model

    name = f"{args.dataset}-_-{model_name.replace('/', '-')}-_-llm_summaries.csv"
    path = Path(args.output) / name

    Path(args.output).mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=True)


main()
'''

