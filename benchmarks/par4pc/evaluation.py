# python benchmarks/par4pc/evaluation.py <path_to_evaluation_results.csv>

import pandas as pd
import argparse
from ast import literal_eval
import numpy as np
from collections import Counter
import csv
from pathlib import Path
import re
from typing import List, Set
import traceback


def summarize_eval(csv_path):
    csv_path = Path(csv_path)
    target_cols = [
        "gold_answer", "silver_answer", "negative_answer",
        "gold_negative", "silver_negative", "negative_negative",
    ]
    counts = Counter({c: 0 for c in target_cols})
    total_rows = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            for col in target_cols:
                try:
                    counts[col] += int(row[col] or 0)
                except (ValueError, KeyError):
                    pass

    pct = {k: round(v / total_rows * 100, 2) if total_rows else 0
           for k, v in counts.items()}
    return counts, pct, total_rows

def parse_answer_string(answer_str: str) -> Set[str]:
    """Parses a comma-separated answer string into a set of uppercase letters."""
    if pd.isna(answer_str) or not isinstance(answer_str, str) or not answer_str:
        return set()
    letters = set(filter(lambda c: 'A' <= c <= 'H', re.split(r'[,\s]+', answer_str.upper())))
    return letters

def main(csv_filepath: str):
    print(f"Analyzing results from: {csv_filepath}")

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    total_data = len(df)
    if total_data == 0:
        print("CSV file is empty. No statistics to calculate.")
        return

    error_count = 0
    if 'error' in df.columns:
        error_count = df['error'].notna().sum()

    exact_match_correct_count = 0
    if 'is_correct' in df.columns:
        if df['is_correct'].dtype == object:
             df['is_correct'] = df['is_correct'].astype(str).str.lower() == 'true'
        exact_match_correct_count = df['is_correct'].sum()

    average_score_percentage = None
    if 'model_score' in df.columns and 'max_score' in df.columns:
        total_model_score_clipped = df['model_score'].clip(lower=0).fillna(0).sum()
        total_max_score = df['max_score'].fillna(0).sum()

        if total_max_score > 0:
            average_score_percentage = (total_model_score_clipped / total_max_score) * 100

    macro_f1 = None
    if all(col in df.columns for col in ['gold_answers', 'predicted_answers', 'error']):
        y_true_metrics = []
        valid_rows = 0
        for _, row in df.iterrows():
            error_val = row['error']
            has_non_llm_error = pd.notna(error_val) and ('LLM call failed' not in str(error_val))
            if has_non_llm_error:
                 continue

            gold_set = parse_answer_string(row['gold_answers'])
            pred_set = parse_answer_string(row['predicted_answers'])
            tp = len(pred_set.intersection(gold_set))
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            y_true_metrics.append({'p': precision, 'r': recall, 'f1': f1})
            valid_rows += 1

        if valid_rows > 0:
            macro_f1 = sum(item['f1'] for item in y_true_metrics) / valid_rows

    df['gold_answer_count'] = df['gold_answers'].apply(lambda x: len(parse_answer_string(x)))

    df_gold_1 = df[df['gold_answer_count'] == 1]
    count_gold_1 = len(df_gold_1)
    score_gold_1_percentage = None
    if count_gold_1 > 0 and all(col in df_gold_1.columns for col in ['model_score', 'max_score']):
        total_model_score_1 = df_gold_1['model_score'].clip(lower=0).fillna(0).sum()
        total_max_score_1 = df_gold_1['max_score'].fillna(0).sum()
        if total_max_score_1 > 0:
            score_gold_1_percentage = (total_model_score_1 / total_max_score_1) * 100

    df_gold_2_5 = df[(df['gold_answer_count'] >= 2) & (df['gold_answer_count'] <= 5)]
    count_gold_2_5 = len(df_gold_2_5)
    score_gold_2_5_percentage = None
    if count_gold_2_5 > 0 and all(col in df_gold_2_5.columns for col in ['model_score', 'max_score']):
        total_model_score_2_5 = df_gold_2_5['model_score'].clip(lower=0).fillna(0).sum()
        total_max_score_2_5 = df_gold_2_5['max_score'].fillna(0).sum()
        if total_max_score_2_5 > 0:
            score_gold_2_5_percentage = (total_model_score_2_5 / total_max_score_2_5) * 100

    print(f"\n--- Evaluation Metrics --- ")

    if average_score_percentage is not None:
        print(f"1. Custom Score (Overall Percentage): {average_score_percentage:.2f}%")
    else:
        print("1. Custom Score (Overall Percentage): N/A (Missing score columns or max score is zero)")

    if macro_f1 is not None:
        print(f"2. Macro F1-Score (Gold):           {macro_f1:.4f}")
    else:
        print("2. Macro F1-Score (Gold):           N/A (Missing required columns or no valid rows)")

    print(f"3. Count (Gold Answers == 1):         {count_gold_1}")
    if score_gold_1_percentage is not None:
        print(f"4. Custom Score (Gold Answers == 1):  {score_gold_1_percentage:.2f}%")
    else:
        print(f"4. Custom Score (Gold Answers == 1):  N/A (No matching cases or missing score data)")

    print(f"5. Count (2 <= Gold Answers <= 5):    {count_gold_2_5}")
    if score_gold_2_5_percentage is not None:
        print(f"6. Custom Score (2 <= Gold Ans <= 5): {score_gold_2_5_percentage:.2f}%")
    else:
        print(f"6. Custom Score (2 <= Gold Ans <= 5): N/A (No matching cases or missing score data)")

    cnt, pct, n = summarize_eval(csv_filepath)
    print("\n=== 📊 Count summary ===============================")
    print(f"Total records: {n}\n")
    print("Item                 Count    Percent(%)")
    print("---------------------------------------------------------")
    for k in cnt:
        print(f"{k:<18} {cnt[k]:>5}   {pct[k]:>6}%")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze evaluation results CSV file and report key metrics.')
    parser.add_argument('csv_filepath', type=str,
                        help='Path to the evaluation_results.csv file')

    args = parser.parse_args()
    main(args.csv_filepath)