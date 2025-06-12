import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import re

def clean_summary(text: str) -> str:
    """Clean and format summary text."""
    # Remove special tokens if present
    text = re.sub(r'<s>|</s>', '', text)
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def add_attribution(text: str, attribution_phrases: List[str]) -> str:
    """Add attribution to a summary if missing."""
    text_lower = text.lower()
    if not any(phrase in text_lower for phrase in attribution_phrases):
        # Check if text starts with common review patterns
        review_starts = ['pros:', 'cons:', 'strength:', 'weakness:', '-']
        if any(text.lower().startswith(start.lower()) for start in review_starts):
            return f"The review notes that {text[0].lower()}{text[1:]}"
        return f"The paper {text[0].lower()}{text[1:]}"
    return text

def truncate_at_sentence(text: str, max_length: int, min_length: int) -> str:
    """Truncate text at sentence boundary."""
    if len(text) <= max_length:
        return text
        
    # Try to find a sentence boundary
    last_period = text[:max_length].rfind('.')
    if last_period > min_length:
        return text[:last_period + 1]
    
    # If no good sentence boundary, truncate with ellipsis
    return text[:max_length] + "..."

def generate_glimpse_summaries(
    rsa_scores_dict: Dict,
    n_unique_sentences: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate GLIMPSE-Speaker and GLIMPSE-Unique summaries from RSA scores.
    
    Args:
        rsa_scores_dict (Dict): Dictionary containing RSA scores
        n_unique_sentences (int): Number of sentences for GLIMPSE-Unique
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: GLIMPSE-Speaker and GLIMPSE-Unique summaries
    """
    rows_speaker = []
    rows_unique = []
    
    # Parameters for better summaries
    min_length = 50
    max_length = 300
    attribution_phrases = [
        'the paper', 'this work', 'the authors', 'they', 'their',
        'this study', 'the study', 'the research', 'the researchers',
        'the method', 'the approach', 'the model', 'the system'
    ]
    
    for result in tqdm(rsa_scores_dict['results']):
        doc_id = result['id'][0]
        best_rsa = result.get('best_rsa', [])
        consensuality_scores = result.get('consensuality_scores', None)
        gold_text = result.get('gold', '')
        
        # Generate GLIMPSE-Speaker summary
        if isinstance(best_rsa, (np.ndarray, list)) and len(best_rsa) > 0:
            candidates = []
            
            # Process top RSA-scored summaries
            for summary in best_rsa[:5]:
                summary = clean_summary(summary)
                
                # Apply length constraints
                if len(summary) < min_length:
                    continue
                
                # Clean and format summary
                summary = truncate_at_sentence(summary, max_length, min_length)
                summary = add_attribution(summary, attribution_phrases)
                
                # Ensure summary ends with proper punctuation
                if not summary.endswith(('.', '!', '?', '...')):
                    summary += '.'
                
                candidates.append((summary, len(summary.split())))
            
            # Select best candidate based on length and completeness
            if candidates:
                # Sort by number of words to prefer more complete summaries
                candidates.sort(key=lambda x: x[1], reverse=True)
                summary = candidates[0][0]
                
                rows_speaker.append({
                    "id": doc_id,
                    "Method": "GLIMPSE-Speaker",
                    "summary": summary,
                    "text": gold_text
                })
        
        # Generate GLIMPSE-Unique summary
        if isinstance(consensuality_scores, pd.Series) and not consensuality_scores.empty:
            # Get most unique sentences
            n_sentences = min(n_unique_sentences, len(consensuality_scores))
            unique_sentences = consensuality_scores.sort_values(ascending=True).head(n_sentences).index.tolist()
            
            # Clean and join sentences
            cleaned_sentences = [clean_summary(sent) for sent in unique_sentences]
            summary = " ".join(cleaned_sentences)
            
            # Add attribution and format
            if not any(phrase in summary.lower() for phrase in attribution_phrases):
                summary = "The paper presents several key points: " + summary
            
            # Apply length constraints
            summary = truncate_at_sentence(summary, max_length, min_length)
            
            # Ensure proper punctuation
            if not summary.endswith(('.', '!', '?', '...')):
                summary += '.'
            
            if len(summary) >= min_length:
                rows_unique.append({
                    "id": doc_id,
                    "Method": "GLIMPSE-Unique",
                    "summary": summary,
                    "text": gold_text
                })
    
    # Convert to DataFrames
    speaker_df = pd.DataFrame(rows_speaker)
    unique_df = pd.DataFrame(rows_unique)
    
    # Print statistics
    print("\nSummary Statistics:")
    print(f"GLIMPSE-Speaker summaries: {len(speaker_df)}")
    print(f"GLIMPSE-Unique summaries: {len(unique_df)}")
    print(f"Average speaker summary length: {speaker_df['summary'].str.len().mean():.2f} chars")
    print(f"Average unique summary length: {unique_df['summary'].str.len().mean():.2f} chars")
    
    return speaker_df, unique_df

def save_summaries(
    speaker_df: pd.DataFrame,
    unique_df: pd.DataFrame,
    output_dir: str,
    year: str = "2017"
) -> None:
    """Save GLIMPSE summaries to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with more descriptive names
    speaker_path = output_dir / f"speaker_summaries_{year}.csv"
    unique_path = output_dir / f"unique_summaries_{year}.csv"
    
    speaker_df.to_csv(speaker_path, index=False)
    unique_df.to_csv(unique_path, index=False)
    
    print(f"\nSaved summaries to:")
    print(f"- {speaker_path}")
    print(f"- {unique_path}")

def process_rsa_scores(
    rsa_scores_path: str,
    output_dir: str,
    year: str = "2017",
    n_unique_sentences: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process RSA scores to generate and save GLIMPSE summaries."""
    print(f"Loading RSA scores from {rsa_scores_path}...")
    rsa_scores_dict = pd.read_pickle(rsa_scores_path)
    
    print(f"\nGenerating summaries for {len(rsa_scores_dict['results'])} documents...")
    speaker_df, unique_df = generate_glimpse_summaries(
        rsa_scores_dict,
        n_unique_sentences=n_unique_sentences
    )
    
    save_summaries(speaker_df, unique_df, output_dir, year)
    return speaker_df, unique_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GLIMPSE summaries from RSA scores")
    parser.add_argument("--rsa_scores", type=str, required=True,
                      help="Path to RSA scores pickle file")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for summaries")
    parser.add_argument("--year", type=str, default="2017",
                      help="Year of the dataset")
    parser.add_argument("--n_unique_sentences", type=int, default=3,
                      help="Number of sentences in GLIMPSE-Unique summaries")
    
    args = parser.parse_args()
    
    process_rsa_scores(
        args.rsa_scores,
        args.output_dir,
        year=args.year,
        n_unique_sentences=args.n_unique_sentences
    )