def evaluate_seahorse_metrics(summaries_path: str, device: str = "cuda") -> None:
    """
    Evaluate SEAHORSE metrics for a summary file.
    
    Args:
        summaries_path (str): Path to summaries CSV file
        device (str): Device to use for evaluation (default: "cuda")
    """
    print(f"\nEvaluating SEAHORSE metrics for {summaries_path}")
    
    # For SEAHORSE metrics, we need 'text' and 'summary' columns
    df = pd.read_csv(summaries_path)
    if 'gold' in df.columns and 'text' not in df.columns:
        df['text'] = df['gold']
        df = df[['text', 'summary']]
        df.to_csv(summaries_path, index=False)
    
    if not all(col in df.columns for col in ['text', 'summary']):
        raise ValueError(f"File {summaries_path} does not have required columns 'text' and 'summary'")
    
    questions = ["1", "2", "3", "4", "5", "6"]  # All SEAHORSE questions
    
    for question in questions:
        print(f"Evaluating question {question}")
        cmd = [
            "python", "glimpse/evaluate/evaluate_seahorse_metrics_samples.py",
            "--summaries", summaries_path,
            "--question", question,
            "--device", device,
            "--batch_size", "8"  # Reduced batch size to handle memory better
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error output:")
            print(result.stderr)
            print("\nWarning: SEAHORSE metrics evaluation failed. This might be due to:")
            print("1. Model not being available or accessible")
            print("2. CUDA memory issues")
            print("3. Network connectivity issues")
            print("\nContinuing with other metrics...")
            continue  # Continue with next question instead of raising error