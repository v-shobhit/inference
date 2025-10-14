#!/usr/bin/env python3
"""
Simplified evaluation script for HLE with structured output format.

Since we now instruct the model to use 'assistantfinalAnswer: <answer>',
we can simplify extraction with fewer pattern matching rules.
"""

import argparse
import logging
import pickle
import re
from typing import Optional, Tuple
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_answer(text: str, answer_type: str) -> Optional[str]:
    """
    Parse answer using simplified patterns focused on 'assistantfinal' marker.
    
    Since we instruct the model to use 'assistantfinalAnswer: <answer>',
    we prioritize that pattern and have streamlined fallbacks for truncated outputs.
    """
    if not text or pd.isna(text):
        return None
    
    text = str(text).strip()
    
    # Primary pattern: assistantfinal (with or without "Answer:")
    # Handles: assistantfinalAnswer: X, assistantfinal X, assistantfinalassistantfinalAnswer: X
    pattern = r'assistantfinal(?:assistant)?(?:final)?(?:Answer)?[:=]?\s*(.+?)(?:\n|$)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if matches:
        answer = matches[-1].group(1).strip()
        
        # Clean up the answer
        answer = answer.strip('*.,;:')
        
        # For multiple choice, extract just the letter
        if answer_type == 'multipleChoice':
            # Extract first capital letter
            letter_match = re.search(r'\b([A-Z])\b', answer)
            if letter_match:
                return letter_match.group(1)
        
        return answer
    
    # Fallback patterns for truncated outputs
    
    # Fallback 1: "Thus final answer: X" or "Thus answer: X"
    pattern_thus = r'\b(?:thus|so|therefore)\s+(?:final\s+)?answer\s*[:=]?\s*["\']?(.+?)(?:["\']?\s*(?:\n|$))'
    matches = list(re.finditer(pattern_thus, text, re.IGNORECASE))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = answer.strip('*.,;:"\'"')
        
        if answer_type == 'multipleChoice':
            letter_match = re.search(r'\b([A-Z])\b', answer)
            if letter_match:
                return letter_match.group(1)
        
        # For exactMatch, limit length to avoid grabbing long explanations
        if len(answer) < 200:
            return answer
    
    # Fallback 2: "Answer: X" or "answer is X"
    pattern_answer = r'\b[Aa]nswer\s*(?:is|:|=)\s*["\']?(.+?)(?:["\']?\s*(?:\n|$))'
    matches = list(re.finditer(pattern_answer, text))
    if matches:
        answer = matches[-1].group(1).strip()
        answer = answer.strip('*.,;:"\'"')
        
        if answer_type == 'multipleChoice':
            letter_match = re.search(r'\b([A-Z])\b', answer)
            if letter_match:
                return letter_match.group(1)
        
        if len(answer) < 200:
            return answer
    
    # Fallback 3: For exactMatch, look for \boxed{} 
    if answer_type == 'exactMatch':
        pattern_boxed = r'\\boxed\{(.+?)\}'
        matches = list(re.finditer(pattern_boxed, text))
        if matches:
            return matches[-1].group(1).strip()
    
    # Fallback 4: For multipleChoice, look for standalone letter at end
    if answer_type == 'multipleChoice':
        # Last standalone capital letter in last 500 chars
        tail = text[-500:] if len(text) > 500 else text
        pattern_letter = r'\b([A-Z])\b(?!.*\b[A-Z]\b)'  # Last capital letter
        match = re.search(pattern_letter, tail)
        if match:
            return match.group(1)
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer or pd.isna(answer):
        return ""
    
    answer = str(answer).strip()
    answer = ' '.join(answer.split())  # Normalize whitespace
    answer = answer.lower()
    
    return answer


def evaluate_answer(extracted: Optional[str], ground_truth: str, answer_type: str) -> bool:
    """Evaluate if extracted answer matches ground truth."""
    if not extracted or pd.isna(extracted):
        return False
    
    if not ground_truth or pd.isna(ground_truth):
        return False
    
    if answer_type == 'multipleChoice':
        # Simple uppercase comparison
        extracted = str(extracted).strip().upper()
        ground_truth = str(ground_truth).strip().upper()
        return extracted == ground_truth
    
    else:  # exactMatch
        # Normalize both for comparison
        extracted_norm = normalize_answer(extracted)
        ground_truth_norm = normalize_answer(ground_truth)
        
        if not extracted_norm or not ground_truth_norm:
            return False
        
        # Exact match after normalization
        if extracted_norm == ground_truth_norm:
            return True
        
        # Partial match with length ratio check (for cases with multiple valid forms)
        if extracted_norm in ground_truth_norm or ground_truth_norm in extracted_norm:
            len_ratio = min(len(extracted_norm), len(ground_truth_norm)) / max(len(extracted_norm), len(ground_truth_norm))
            if len_ratio > 0.5:
                return True
        
        return False


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process entire dataframe and add evaluation columns."""
    
    # Validate required columns
    required_cols = ['model_output', 'answer', 'answer_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Processing {len(df)} rows...")
    
    # Create output dataframe
    df_output = df.copy()
    
    # Process each row
    extracted_answers = []
    accuracies = []
    
    for idx, row in df_output.iterrows():
        answer_type = str(row['answer_type']).strip()
        model_output = row['model_output']
        ground_truth = row['answer']
        
        # Parse answer
        extracted = parse_answer(model_output, answer_type)
        extracted_answers.append(extracted)
        
        # Evaluate
        is_correct = evaluate_answer(extracted, ground_truth, answer_type)
        accuracies.append(1.0 if is_correct else 0.0)
    
    # Add results to dataframe
    df_output['extracted_answer'] = extracted_answers
    df_output['accuracy'] = accuracies
    
    return df_output


def print_statistics(df_evaluated: pd.DataFrame):
    """Print evaluation statistics."""
    
    total = len(df_evaluated)
    
    # Calculate answered (extraction success rate)
    answered_count = df_evaluated['extracted_answer'].notna().sum()
    answered_fraction = answered_count / total if total > 0 else 0.0
    
    # Calculate accuracy
    correct_count = (df_evaluated['accuracy'] == 1.0).sum()
    accuracy = correct_count / total if total > 0 else 0.0
    accuracy_on_answered = correct_count / answered_count if answered_count > 0 else 0.0
    
    # Calculate by answer type
    for answer_type in df_evaluated['answer_type'].unique():
        subset = df_evaluated[df_evaluated['answer_type'] == answer_type]
        subset_total = len(subset)
        subset_answered = subset['extracted_answer'].notna().sum()
        subset_correct = (subset['accuracy'] == 1.0).sum()
        subset_answered_fraction = subset_answered / subset_total if subset_total > 0 else 0.0
        subset_accuracy = subset_correct / subset_total if subset_total > 0 else 0.0
        
        logger.info(f"\n{answer_type}:")
        logger.info(f"  Total: {subset_total}")
        logger.info(f"  Answered: {subset_answered} ({subset_answered_fraction:.2%})")
        logger.info(f"  Correct: {subset_correct}")
        logger.info(f"  Accuracy: {subset_accuracy:.2%}")
    
    # Mean token length
    if 'tok_model_output_len' in df_evaluated.columns:
        mean_tokens = df_evaluated['tok_model_output_len'].mean()
        logger.info(f"\nMean output tokens: {mean_tokens:.1f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Answered (extracted): {answered_count} ({answered_fraction:.2%})")
    print(f"Correct: {correct_count}")
    print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{total})")
    print(f"Accuracy on answered: {accuracy_on_answered:.2%} ({correct_count}/{answered_count})")
    
    if 'tok_model_output_len' in df_evaluated.columns:
        print(f"Mean output tokens: {mean_tokens:.1f}")
    
    print("=" * 80)
    
    return {
        'total': total,
        'answered': answered_count,
        'answered_fraction': answered_fraction,
        'correct': correct_count,
        'accuracy': accuracy,
        'accuracy_on_answered': accuracy_on_answered,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simplified evaluation for HLE with structured output")
    parser.add_argument("--input-file", required=True,
                        help="Input pickle file with model outputs")
    parser.add_argument("--output-file",
                        help="Output pickle file (defaults to <input>_evaluated.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / (input_path.stem + "_evaluated.pkl")
    
    logger.info(f"Loading data from: {input_path}")
    
    # Load data
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input file must contain a pandas DataFrame")
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Process dataframe
    df_evaluated = process_dataframe(df)
    
    # Print statistics
    stats = print_statistics(df_evaluated)
    
    # Save results
    logger.info(f"Saving evaluated results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(df_evaluated, f)
    
    logger.info("Done!")
    
    return stats


if __name__ == "__main__":
    main()
