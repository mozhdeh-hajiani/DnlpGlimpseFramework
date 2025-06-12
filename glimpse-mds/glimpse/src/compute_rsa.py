from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm
from pickle import dump
import torch
import sys
import os

# Add the parent directory of `rsasumm` to the Python path
sys.path.append('/content/drive/MyDrive/glimpse/glimpse-mds/')

from rsasumm.rsa_reranker import RSAReranking

DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a pickle file.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def parse_summaries(path: Path) -> pd.DataFrame:
    try:
        summaries = pd.read_csv(path)
        print(f"Loaded DataFrame with columns: {summaries.columns.tolist()}")
        
        # Remove duplicate id column if it exists
        if 'id.1' in summaries.columns:
            summaries = summaries.drop('id.1', axis=1)
            print("Removed duplicate 'id.1' column")
        
        # Check if 'id' column exists, if not add it
        if 'id' not in summaries.columns:
            summaries['id'] = summaries.index + 1
            print("Added 'id' column")
            
        # Ensure all required columns exist
        required_columns = ["index", "id", "text", "gold", "summary", "id_candidate"]
        missing_columns = [col for col in required_columns if col not in summaries.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert text columns to string type
        text_columns = ['summary', 'text', 'gold']
        for col in text_columns:
            if col in summaries.columns:
                summaries[col] = summaries[col].astype(str)
                print(f"Converted {col} column to string type")
            
        # Clean the data
        summaries = summaries.dropna(subset=['summary', 'text'])  # Remove rows with missing summaries or texts
        
        # Remove empty strings and whitespace-only strings
        summaries = summaries[summaries['summary'].str.strip() != '']
        summaries = summaries[summaries['text'].str.strip() != '']
        
        # Remove any rows where summary or text is 'nan' (as string)
        summaries = summaries[~summaries['summary'].str.lower().isin(['nan', 'none'])]
        summaries = summaries[~summaries['text'].str.lower().isin(['nan', 'none'])]
        
        print(f"Cleaned DataFrame shape: {summaries.shape}")
        print(f"Data types after cleaning:")
        for col in text_columns:
            if col in summaries.columns:
                print(f"{col}: {summaries[col].dtype}")
        
        return summaries
        
    except Exception as e:
        raise ValueError(f"Error loading dataset {path}: {str(e)}")

def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device):
    results = []
    
    # Set a reasonable max length
    max_length = min(tokenizer.model_max_length, 1024)  # Cap at 1024
    print(f"Using max_length: {max_length}")

    for name, group in tqdm(summaries.groupby(["id"])):
        try:
            # Get unique candidates and source texts
            candidates = group.summary.unique().tolist()
            source_texts = group.text.unique().tolist()
            
            # Filter out None or empty strings
            candidates = [c for c in candidates if c and isinstance(c, str)]
            source_texts = [t for t in source_texts if t and isinstance(t, str)]
            
            if not candidates or not source_texts:
                print(f"Skipping group {name} due to empty candidates or source texts")
                continue
                
            # Truncate candidates and source texts safely
            truncated_candidates = []
            truncated_source_texts = []
            
            for c in candidates:
                try:
                    encoded = tokenizer.encode(c, max_length=max_length, truncation=True)
                    decoded = tokenizer.decode(encoded)
                    truncated_candidates.append(decoded)
                except Exception as e:
                    print(f"Error processing candidate: {str(e)}")
                    continue
                    
            for t in source_texts:
                try:
                    encoded = tokenizer.encode(t, max_length=max_length, truncation=True)
                    decoded = tokenizer.decode(encoded)
                    truncated_source_texts.append(decoded)
                except Exception as e:
                    print(f"Error processing source text: {str(e)}")
                    continue
            
            if not truncated_candidates or not truncated_source_texts:
                print(f"Skipping group {name} due to empty truncated texts")
                continue

            rsa_reranker = RSAReranking(
                model,
                tokenizer,
                device=device,
                candidates=truncated_candidates,
                source_texts=truncated_source_texts,
                batch_size=8, #8
                rationality=3,
            )
            
            (
                best_rsa,
                best_base,
                speaker_df,
                listener_df,
                initial_listener,
                language_model_proba_df,
                initial_consensuality_scores,
                consensuality_scores,
            ) = rsa_reranker.rerank(t=2) # = rsa_reranker.rerank(t=2)

            gold = group['gold'].iloc[0]  # Use iloc instead of tolist()[0]

            results.append(
                {
                    "id": name,
                    "best_rsa": best_rsa,
                    "best_base": best_base,
                    "speaker_df": speaker_df,
                    "listener_df": listener_df,
                    "initial_listener": initial_listener,
                    "language_model_proba_df": language_model_proba_df,
                    "initial_consensuality_scores": initial_consensuality_scores,
                    "consensuality_scores": consensuality_scores,
                    "gold": gold,
                    "rationality": 3,
                    "text_candidates": group
                }
            )
            
        except Exception as e:
            print(f"Error processing group {name}: {str(e)}")
            continue
            
    return results

def main():
    args = parse_args()

    if args.filter is not None and args.filter not in args.summaries.stem:
        return

    try:
        # Load model and tokenizer
        print(f"Loading model: {args.model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        
        if "pegasus" in args.model_name:
            tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Set tokenizer max length
        tokenizer.model_max_length = 1024
        print(f"Tokenizer max_length: {tokenizer.model_max_length}")

        # Move model to device
        model = model.to(args.device)
        print(f"Model loaded on {args.device}")

        # Load and parse summaries
        summaries = parse_summaries(args.summaries)
        print(f"Loaded {len(summaries)} summaries.")

        # Compute RSA
        print("Computing RSA matrices...")
        results = compute_rsa(summaries, model, tokenizer, args.device)

        # Prepare results
        results = {"results": results}
        results["metadata/reranking_model"] = args.model_name
        results["metadata/rsa_iterations"] = 3

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.summaries.stem}-_-r3-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
        
        with open(output_path, "wb") as f:
            dump(results, f)
        print(f"Results saved to {output_path}")

        if args.scripted_run:
            print(output_path)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()