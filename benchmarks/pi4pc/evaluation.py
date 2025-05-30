# python benchmarks/pi4pc/evaluation.py <path_to_evaluation_results.csv>

import pandas as pd
import argparse
import numpy as np
import traceback
from pathlib import Path
from collections import Counter
import csv

def summarize_eval(csv_path):
    csv_path = Path(csv_path)
    target_cols = [
        "gold_answer", "silver_answer", "negative_answer",
        "gold_negative", "silver_negative", "negative_negative",
    ]
    counts = Counter({c: 0 for c in target_cols})
    total_rows = 0

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            try:
                first_col_name = reader.fieldnames[0] if reader.fieldnames else None
            except IndexError:
                first_col_name = None

            if not first_col_name:
                print("Warning in summarize_eval: Could not determine the first column name. Skipping TOTAL row check.")
                first_col_name = '__DUMMY__'

            for row in reader:
                if str(row.get(first_col_name, '')).strip().upper().startswith('TOTAL'):
                    continue

                total_rows += 1
                for col in target_cols:
                    try:
                        value_str = row.get(col, '0')
                        counts[col] += int(value_str) if value_str else 0
                    except (ValueError, KeyError):
                        pass

    except FileNotFoundError:
        print(f"Error in summarize_eval: File not found at {csv_path}")
        return Counter(), {}, 0
    except Exception as e:
        print(f"Error reading or processing CSV in summarize_eval: {e}")
        print(traceback.format_exc())
        return Counter(), {}, 0

    total_count_sum = sum(counts.values())

    pct = {k: round(v / total_count_sum * 100, 2) if total_count_sum > 0 else 0
           for k, v in counts.items()}

    return counts, pct, total_rows


def main(csv_filepath: str):
    print(f"Analyzing results from: {csv_filepath}")

    try:
        df = pd.read_csv(csv_filepath, dtype={'is_valid_option_key': 'object'})

        if not df.empty and df.columns.any():
            first_col_name = df.columns[0]
            df = df[~df[first_col_name].astype(str).str.upper().str.startswith('TOTAL', na=False)]

        if 'is_valid_option_key' in df.columns:
            df['is_valid_option_key'] = df['is_valid_option_key'].fillna('False').astype(str).str.lower() == 'true'
        else:
            print("Warning: 'is_valid_option_key' column not found. Cannot count invalid responses.")
            df['is_valid_option_key'] = True

    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print(traceback.format_exc())
        return

    total_items = len(df)
    if total_items == 0:
        print("CSV file is empty. No statistics to calculate.")
        return

    print(f"Total items found in CSV: {total_items}")

    error_count = 0
    if 'error' in df.columns:
        error_count = df['error'].notna().sum()
        print(f"Number of items with processing errors: {error_count}")
    else:
        print("Warning: 'error' column not found.")

    successful_df = df[df['error'].isna()] if 'error' in df.columns else df
    successful_count = len(successful_df)

    invalid_response_count = 0
    if 'is_valid_option_key' in successful_df.columns:
        invalid_response_count = (~successful_df['is_valid_option_key']).sum()
        print(f"Number of items with invalid responses (among successful): {invalid_response_count}")

    if successful_count == 0:
        print("No successfully processed items found. Cannot calculate performance metrics.")
        counts, percentages, total_rows_in_summary = summarize_eval(csv_filepath)
        print("\n=== 📊 Count summary ===============================")
        print(f"Total records processed by summarize_eval: {total_rows_in_summary}\n")
        print("Item                Count    Ratio(%)")
        print("---------------------------------------------------------")
        for k in counts:
            print(f"{k:<18} {counts[k]:>5}   {percentages[k]:>6}%")
        print("=========================================================\n")
        return

    print(f"Number of successfully processed items (used for metrics): {successful_count}")

    counts, percentages, total_rows_in_summary = summarize_eval(csv_filepath)

    tp = counts.get('gold_answer', 0) + counts.get('silver_answer', 0)
    fp = counts.get('negative_answer', 0)
    fn = counts.get('gold_negative', 0) + counts.get('silver_negative', 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    gold_answers = counts.get('gold_answer', 0)
    silver_answers = counts.get('silver_answer', 0)
    custom_score = (2 * gold_answers + 1 * silver_answers) / successful_count if successful_count > 0 else 0

    print(f"\n--- Evaluation Metrics (Calculated from Counts) --- ")

    print(f"1. Precision:             {precision:.4f} (TP={tp}, FP={fp})")
    print(f"2. Recall:                {recall:.4f} (TP={tp}, FN={fn})")
    print(f"3. F1-Score:              {f1_score:.4f}")
    print(f"4. Custom Score:          {custom_score:.4f} (Weighted Acc: (2*Gold + 1*Silver) / Success)")

    error_rate = (error_count / total_items) * 100 if total_items > 0 else 0
    invalid_rate_overall = (invalid_response_count / total_items) * 100 if total_items > 0 else 0
    invalid_rate_successful = (invalid_response_count / successful_count) * 100 if successful_count > 0 else 0

    print(f"\n--- Additional Stats --- ")
    print(f"Overall Error Rate:           {error_rate:.2f}% ({error_count}/{total_items})")
    print(f"Invalid Response Rate (vs All): {invalid_rate_overall:.2f}% ({invalid_response_count}/{total_items})")
    print(f"Invalid Response Rate (vs Success): {invalid_rate_successful:.2f}% ({invalid_response_count}/{successful_count})")

    print("\n=== 📊 Count summary ===============================")
    print(f"Total records processed by summarize_eval: {total_rows_in_summary}\n")
    print("Item                Count    Ratio(%)")
    print("---------------------------------------------------------")
    for k in counts:
        print(f"{k:<18} {counts[k]:>5}   {percentages[k]:>6}%")
    print("=========================================================\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze evaluation results CSV file for the cited paragraph task.')
    parser.add_argument('csv_filepath', type=str,
                        help='Path to the evaluation_results.csv file generated by pi4pc/test.py')

    args = parser.parse_args()
    
    main(args.csv_filepath)
