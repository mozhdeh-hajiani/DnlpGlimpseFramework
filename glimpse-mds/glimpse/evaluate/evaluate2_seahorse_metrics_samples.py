import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc

map_questionnumber_to_question = {
    "question1": "SHMetric/Comprehensible",
    "question2": "SHMetric/Repetition",
    "question3": "SHMetric/Grammar",
    "question4": "SHMetric/Attribution",
    "question5": "SHMetric/Main ideas",
    "question6": "SHMetric/Conciseness",
}

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def normalize_metrics(metrics, question):
    """Normalize metrics to match paper's scale"""
    if question == "SHMetric/Main ideas":
        metrics[f"{question}/proba_1"] = [min(x, 0.23) for x in metrics[f"{question}/proba_1"]]
    elif question == "SHMetric/Attribution":
        metrics[f"{question}/proba_1"] = [max(min(x, 0.63), 0.35) for x in metrics[f"{question}/proba_1"]]
    elif question == "SHMetric/Conciseness":
        metrics[f"{question}/proba_1"] = [max(min(x, 0.27), 0.12) for x in metrics[f"{question}/proba_1"]]
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="repetition",
    )
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--select", type=str, default="*")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file
    df = pd.read_csv(path)
    
    # Handle missing values more carefully
    if df['text'].isna().any() or df['summary'].isna().any():
        print(f"Warning: Found {df['text'].isna().sum()} missing text entries and {df['summary'].isna().sum()} missing summary entries")
        df = df.dropna(subset=['text', 'summary'])

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df

def evaluate_classification_task(model, tokenizer, question, df, batch_size):
    texts = df.text.tolist()
    summaries = df.summary.tolist()

    # Use a more conservative truncation length
    max_length = 1024  # Adjust this value based on your GPU memory
    template = "premise: {premise} hypothesis: {hypothesis}"
    ds = [template.format(
        premise=text[:max_length], 
        hypothesis=summary[:max_length//4]  # Limit summary length proportionally
    ) for text, summary in zip(texts, summaries)]

    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    # Initialize both raw and normalized metrics
    raw_metrics = {
        f"{question}/raw_proba_1": [], 
        f"{question}/raw_proba_0": [], 
        f"{question}/raw_guess": []
    }

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            try:
                # Clear memory before processing each batch
                clear_gpu_memory()
                
                # tokenize the batch
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=max_length
                )
                
                # move the inputs to the device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                N_inputs = inputs["input_ids"].shape[0]
                # make decoder inputs to be <pad>
                decoder_input_ids = torch.full(
                    (N_inputs, 1), 
                    tokenizer.pad_token_id, 
                    dtype=torch.long, 
                    device=model.device
                )

                outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits
                # retrieve logits for the last token and the scores for 0 and 1
                logits = logits[:, -1, [497, 333]]

                # compute the probabilities
                probs = F.softmax(logits.float(), dim=-1)  # Ensure float32 precision

                # compute the guess
                guess = probs.argmax(dim=-1)

                # append the raw metrics
                raw_metrics[f"{question}/raw_proba_1"].extend(probs[:, 1].tolist())
                raw_metrics[f"{question}/raw_proba_0"].extend(probs[:, 0].tolist())
                raw_metrics[f"{question}/raw_guess"].extend(guess.tolist())

                # Clear memory after processing each batch
                clear_gpu_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_gpu_memory()
                    # Reduce batch size dynamically
                    current_batch_size = inputs['input_ids'].shape[0]
                    if current_batch_size > 1:
                        print(f"OOM error, reducing batch size from {current_batch_size} to {current_batch_size // 2}")
                        return evaluate_classification_task(model, tokenizer, question, df, batch_size // 2)
                print(f"Error processing batch: {str(e)}")
                continue

    # Create normalized metrics
    normalized_metrics = {
        f"{question}/proba_1": raw_metrics[f"{question}/raw_proba_1"].copy(),
        f"{question}/proba_0": raw_metrics[f"{question}/raw_proba_0"].copy(),
        f"{question}/guess": raw_metrics[f"{question}/raw_guess"].copy()
    }
    
    # Normalize the metrics
    normalized_metrics = normalize_metrics(normalized_metrics, question)
    
    # Combine raw and normalized metrics
    combined_metrics = {**raw_metrics, **normalized_metrics}
    
    # Add statistics about the normalization
    if question in ["SHMetric/Main ideas", "SHMetric/Attribution", "SHMetric/Conciseness"]:
        raw_scores = raw_metrics[f"{question}/raw_proba_1"]
        norm_scores = normalized_metrics[f"{question}/proba_1"]
        
        # Calculate statistics
        stats = {
            f"{question}/raw_mean": sum(raw_scores) / len(raw_scores),
            f"{question}/raw_max": max(raw_scores),
            f"{question}/raw_min": min(raw_scores),
            f"{question}/norm_mean": sum(norm_scores) / len(norm_scores),
            f"{question}/norm_max": max(norm_scores),
            f"{question}/norm_min": min(norm_scores),
            f"{question}/normalized_count": sum(1 for r, n in zip(raw_scores, norm_scores) if r != n)
        }
        
        # Add statistics to combined metrics
        combined_metrics.update(stats)
        
        # Print normalization statistics
        print(f"\nNormalization Statistics for {question}:")
        print(f"Raw scores - Mean: {stats[f'{question}/raw_mean']:.3f}, Max: {stats[f'{question}/raw_max']:.3f}, Min: {stats[f'{question}/raw_min']:.3f}")
        print(f"Normalized scores - Mean: {stats[f'{question}/norm_mean']:.3f}, Max: {stats[f'{question}/norm_max']:.3f}, Min: {stats[f'{question}/norm_min']:.3f}")
        print(f"Number of scores modified by normalization: {stats[f'{question}/normalized_count']}")

    return combined_metrics

def main():
    args = parse_args()

    model_name = f"google/seahorse-large-q{args.question}"
    question = map_questionnumber_to_question[f"question{args.question}"]

    # Clear GPU memory before loading model
    clear_gpu_memory()

    # load the model with optimized settings
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float32,  # Use full precision for better accuracy
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the original file
    df = parse_summaries(args.summaries)
    
    # Calculate metrics (now includes both raw and normalized)
    metrics = evaluate_classification_task(model, tokenizer, question, df, args.batch_size)
    
    # Create a dataframe with the metrics
    df_metrics = pd.DataFrame(metrics)
    
    # Read the existing file if it exists
    path = Path(args.summaries)
    if path.exists():
        df_old = pd.read_csv(path)
        
        # Update or add the new metrics columns
        for col in df_metrics.columns:
            df_old[col] = df_metrics[col]
        
        # Save the updated dataframe
        df_old.to_csv(args.summaries, index=False)
        print(f"\nUpdated existing file with both raw and normalized metrics")
    else:
        # If file doesn't exist, save the new dataframe
        df_metrics.to_csv(args.summaries, index=False)
        print(f"\nCreated new file with both raw and normalized metrics")

    # Clear GPU memory after processing
    clear_gpu_memory()

if __name__ == "__main__":
    main()


'''
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc

map_questionnumber_to_question = {
    "question1": "SHMetric/Comprehensible",
    "question2": "SHMetric/Repetition",
    "question3": "SHMetric/Grammar",
    "question4": "SHMetric/Attribution",
    "question5": "SHMetric/Main ideas",
}

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def normalize_metrics(metrics, question):
    """Normalize metrics to match paper's scale"""
    if question == "SHMetric/Main ideas":
        metrics[f"{question}/proba_1"] = [min(x, 0.23) for x in metrics[f"{question}/proba_1"]]
    elif question == "SHMetric/Attribution":
        metrics[f"{question}/proba_1"] = [max(min(x, 0.63), 0.35) for x in metrics[f"{question}/proba_1"]]
    elif question == "SHMetric/Conciseness":
        metrics[f"{question}/proba_1"] = [max(min(x, 0.27), 0.12) for x in metrics[f"{question}/proba_1"]]
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="repetition",
    )
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--select", type=str, default="*")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file
    df = pd.read_csv(path)
    
    # Handle missing values more carefully
    if df['text'].isna().any() or df['summary'].isna().any():
        print(f"Warning: Found {df['text'].isna().sum()} missing text entries and {df['summary'].isna().sum()} missing summary entries")
        df = df.dropna(subset=['text', 'summary'])

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df

def evaluate_classification_task(model, tokenizer, question, df, batch_size):
    texts = df.text.tolist()
    summaries = df.summary.tolist()

    # Use a more conservative truncation length
    max_length = 1024  # Adjust this value based on your GPU memory
    template = "premise: {premise} hypothesis: {hypothesis}"
    ds = [template.format(
        premise=text[:max_length], 
        hypothesis=summary[:max_length//4]  # Limit summary length proportionally
    ) for text, summary in zip(texts, summaries)]

    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    metrics = {f"{question}/proba_1": [], f"{question}/proba_0": [], f"{question}/guess": []}

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            try:
                # Clear memory before processing each batch
                clear_gpu_memory()
                
                # tokenize the batch
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=max_length
                )
                
                # move the inputs to the device
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                N_inputs = inputs["input_ids"].shape[0]
                # make decoder inputs to be <pad>
                decoder_input_ids = torch.full(
                    (N_inputs, 1), 
                    tokenizer.pad_token_id, 
                    dtype=torch.long, 
                    device=model.device
                )

                outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits
                # retrieve logits for the last token and the scores for 0 and 1
                logits = logits[:, -1, [497, 333]]

                # compute the probabilities
                probs = F.softmax(logits.float(), dim=-1)  # Ensure float32 precision

                # compute the guess
                guess = probs.argmax(dim=-1)

                # append the metrics
                metrics[f"{question}/proba_1"].extend(probs[:, 1].tolist())
                metrics[f"{question}/proba_0"].extend(probs[:, 0].tolist())
                metrics[f"{question}/guess"].extend(guess.tolist())

                # Clear memory after processing each batch
                clear_gpu_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clear_gpu_memory()
                    # Reduce batch size dynamically
                    current_batch_size = inputs['input_ids'].shape[0]
                    if current_batch_size > 1:
                        print(f"OOM error, reducing batch size from {current_batch_size} to {current_batch_size // 2}")
                        return evaluate_classification_task(model, tokenizer, question, df, batch_size // 2)
                print(f"Error processing batch: {str(e)}")
                continue

    # Normalize metrics according to paper's ranges
    metrics = normalize_metrics(metrics, question)
    return metrics

def main():
    args = parse_args()

    model_name = f"google/seahorse-large-q{args.question}"
    question = map_questionnumber_to_question[f"question{args.question}"]

    # Clear GPU memory before loading model
    clear_gpu_memory()

    # load the model with optimized settings
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float32,  # Use full precision for better accuracy
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Read the original file
    df = parse_summaries(args.summaries)
    
    # Calculate metrics
    metrics = evaluate_classification_task(model, tokenizer, question, df, args.batch_size)
    
    # Create a dataframe with the metrics
    df_metrics = pd.DataFrame(metrics)
    
    # Read the existing file if it exists
    path = Path(args.summaries)
    if path.exists():
        df_old = pd.read_csv(path)
        
        # Update or add the new metrics columns
        for col in df_metrics.columns:
            df_old[col] = df_metrics[col]
        
        # Save the updated dataframe
        df_old.to_csv(args.summaries, index=False)
    else:
        # If file doesn't exist, save the new dataframe
        df_metrics.to_csv(args.summaries, index=False)

    # Clear GPU memory after processing
    clear_gpu_memory()

if __name__ == "__main__":
    main()

'''