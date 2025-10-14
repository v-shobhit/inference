#!/usr/bin/env python3
"""
Download and prepare Humanity's Last Exam (HLE) dataset for gpt-oss inference.

This script downloads the HLE dataset from HuggingFace and saves it as a pickle file
in the same format as other datasets used by harmonize_inputs.py and run_infer.py.
"""

import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import HfFolder


def download_and_prepare_hle(output_dir: str = "/home/mlperf_inference_storage/data/deepseek-r1", token: str = None):
    """
    Download HLE dataset and convert to pickle format.
    
    Args:
        output_dir: Directory to save the pickle file
        token: HuggingFace API token (optional, will use cached token if not provided)
    """
    print("Downloading Humanity's Last Exam dataset from HuggingFace...")
    
    # Try to get token from environment or cache if not provided
    if token is None:
        token = os.environ.get('HF_TOKEN') or HfFolder.get_token()
    
    # Load the dataset from HuggingFace
    # Note: You may need to authenticate with HF first if the dataset requires acceptance
    try:
        dataset = load_dataset("cais/hle", split="test", token=token)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nIf you see an authentication error, you may need to:")
        print("1. Log in to HuggingFace: huggingface-cli login")
        print("2. Accept the dataset terms at: https://huggingface.co/datasets/cais/hle")
        print("3. Set HF_TOKEN environment variable or pass --token argument")
        raise
    
    print(f"Loaded {len(dataset)} questions from HLE dataset")
    
    # Convert to pandas DataFrame
    print("Converting to DataFrame format...")
    data_list = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing questions")):
        # HLE dataset structure - inspect what fields are available
        # Typical fields might include: question, answer, choices, subject, etc.
        
        # Build the question text
        question_text = item.get('question', '')
        
        # Handle multiple choice questions - add choices if available
        if 'choices' in item and item['choices']:
            choices_text = "\n".join([f"{chr(65+i)}) {choice}" 
                                     for i, choice in enumerate(item['choices'])])
            full_question = f"{question_text}\n\n{choices_text}"
        else:
            full_question = question_text
        
        # Get the answer
        answer = item.get('answer', '')
        
        # Get subject/topic if available
        subject = item.get('subject', 'general')
        
        # Create a row in the same format as other datasets
        row = {
            'dataset': 'hle',
            'question': full_question,
            'answer': answer,
            'subject': subject,
            'qid': idx,
            # These will be filled by harmonize_inputs.py
            'tok_input': None,
            'tok_input_len': None,
            'text_input': None,
            # Reference fields set to None as per harmonize_inputs.py pattern
            'ref_accuracy': None,
            'ref_extracted_answer': None,
            'ref_output': None,
            'tok_ref_output': None,
            'tok_ref_output_len': None,
        }
        
        # Add any additional fields from the dataset
        for key, value in item.items():
            if key not in row:
                row[key] = value
        
        data_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to pickle
    output_file = os.path.join(output_dir, "hle_dataset.pkl")
    print(f"\nSaving to {output_file}...")
    df.to_pickle(output_file)
    
    print(f"\n✓ Successfully saved HLE dataset:")
    print(f"  - Total questions: {len(df)}")
    print(f"  - Output file: {output_file}")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Print sample question
    print(f"\n--- Sample Question ---")
    print(f"Question: {df.iloc[0]['question'][:200]}...")
    print(f"Answer: {df.iloc[0]['answer']}")
    print(f"Subject: {df.iloc[0]['subject']}")
    
    # Print dataset statistics
    if 'subject' in df.columns:
        print(f"\n--- Subject Distribution ---")
        print(df['subject'].value_counts().head(10))
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare HLE dataset for gpt-oss inference")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/mlperf_inference_storage/data/deepseek-r1",
        help="Directory to save the pickle file (default: /home/mlperf_inference_storage/data/deepseek-r1)"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace API token for accessing gated datasets (optional if already logged in)"
    )
    
    args = parser.parse_args()
    
    try:
        df = download_and_prepare_hle(args.output_dir, token=args.token)
        print("\n✓ Dataset preparation complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

