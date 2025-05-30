# $ python test_citedPatent/spliter.py --input_dir data/mcq_cited_patent --output_base_dir data/benchmark_cited_patent_splited --ratios 90:5:5

import os
import re
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from math import floor
import pandas as pd
import pyarrow


def _save_split_data(data_list: list, output_path_base: Path, split_name: str):
    """Helper function to save aggregated data to jsonl and parquet."""
    if not data_list:
        print(f"Warning: No data found for split '{split_name}'. Skipping file generation.")
        return None, None

    jsonl_path = output_path_base / f"{split_name}.jsonl"
    parquet_path = output_path_base / f"{split_name}.parquet"

    # Save to JSONL
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
            for item in data_list:
                f_jsonl.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {split_name} data to {jsonl_path}")
    except IOError as e:
        print(f"Error saving {split_name} data to {jsonl_path}: {e}")
        jsonl_path = None # Indicate failure

    # Save to Parquet
    try:
        df = pd.DataFrame(data_list)
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully saved {split_name} data to {parquet_path}")
    except Exception as e: # Catch pandas/pyarrow errors
        print(f"Error saving {split_name} data to {parquet_path}: {e}")
        print("Make sure 'pandas' and 'pyarrow' are installed ('pip install pandas pyarrow').")
        parquet_path = None # Indicate failure

    return jsonl_path, parquet_path

def split_data(input_dir: str, output_base_dir: str, ratios: tuple = (0.9, 0.05, 0.05), seed: int = 42):
    """
    Splits MCQ JSON files into train, validation, and test sets based on application number.

    Args:
        input_dir: Directory containing the source mcq_*.json files.
        output_base_dir: Base directory to create train/validation/test subdirectories.
        ratios: Tuple representing the split ratios for train, validation, test.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        return

    mcq_files = list(input_path.glob("mcq_r*_cl*.json"))
    if not mcq_files:
        print(f"Error: No MCQ files found in {input_path}")
        return

    print(f"Found {len(mcq_files)} MCQ files.")

    # Group files by application number
    apps = defaultdict(list)
    app_num_pattern = re.compile(r"mcq_r\d+_(\d+)_cl\d+\.json")

    for file_path in mcq_files:
        match = app_num_pattern.match(file_path.name)
        if match:
            app_num = match.group(1)
            apps[app_num].append(file_path)
        else:
            print(f"Warning: Skipping file with unexpected name format: {file_path.name}")

    if not apps:
        print("Error: Could not extract application numbers from any files.")
        return

    unique_app_nums = list(apps.keys())
    random.shuffle(unique_app_nums)
    total_apps = len(unique_app_nums)
    print(f"Found {total_apps} unique application numbers.")

    train_size = floor(total_apps * ratios[0])
    valid_size = floor(total_apps * ratios[1])
    test_size = total_apps - train_size - valid_size

    if train_size + valid_size + test_size != total_apps:
        print(f"Warning: Split sizes don't perfectly match total due to rounding. Adjusting test set size.")
        pass

    print(f"Splitting applications: Train={train_size}, Validation={valid_size}, Test={test_size}")

    train_apps = unique_app_nums[:train_size]
    valid_apps = unique_app_nums[train_size : train_size + valid_size]
    test_apps = unique_app_nums[train_size + valid_size :]

    train_data, valid_data, test_data = [], [], []
    file_counts = {"train": 0, "validation": 0, "test": 0}

    print("\nProcessing and aggregating data...")

    for app_num in train_apps:
        for file_path in apps[app_num]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        train_data.extend(content)
                    else:
                        train_data.append(content)
                    file_counts["train"] += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    for app_num in valid_apps:
        for file_path in apps[app_num]:
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        valid_data.extend(content)
                    else:
                        valid_data.append(content)
                    file_counts["validation"] += 1
             except json.JSONDecodeError:
                 print(f"Warning: Skipping invalid JSON file: {file_path}")
             except Exception as e:
                 print(f"Error reading file {file_path}: {e}")

    for app_num in test_apps:
        for file_path in apps[app_num]:
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        test_data.extend(content)
                    else:
                        test_data.append(content)
                    file_counts["test"] += 1
             except json.JSONDecodeError:
                 print(f"Warning: Skipping invalid JSON file: {file_path}")
             except Exception as e:
                 print(f"Error reading file {file_path}: {e}")

    print("Data aggregation complete. Saving files...")


    output_base_path.mkdir(parents=True, exist_ok=True)
    train_jsonl, train_parquet = _save_split_data(train_data, output_base_path, "train")
    valid_jsonl, valid_parquet = _save_split_data(valid_data, output_base_path, "validation")
    test_jsonl, test_parquet = _save_split_data(test_data, output_base_path, "test")

    print("\n--- Splitting Summary ---")
    print(f"Input directory: {input_path}")
    print(f"Output base directory: {output_base_path}")
    print("-" * 20)
    print(f"Train set: {len(train_apps)} applications, {file_counts['train']} files processed, {len(train_data)} records")
    if train_jsonl: print(f"  -> {train_jsonl}")
    if train_parquet: print(f"  -> {train_parquet}")
    print("-" * 20)
    print(f"Validation set: {len(valid_apps)} applications, {file_counts['validation']} files processed, {len(valid_data)} records")
    if valid_jsonl: print(f"  -> {valid_jsonl}")
    if valid_parquet: print(f"  -> {valid_parquet}")
    print("-" * 20)
    print(f"Test set: {len(test_apps)} applications, {file_counts['test']} files processed, {len(test_data)} records")
    if test_jsonl: print(f"  -> {test_jsonl}")
    if test_parquet: print(f"  -> {test_parquet}")
    print("------------------------")

def main():
    parser = argparse.ArgumentParser(description="Split MCQ benchmark files into train/validation/test sets based on application number, saving aggregated data as jsonl and parquet.")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='dataset_generator/data/mcq_cited_patent',
        help='Directory containing the source mcq_*.json files.'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='data_split',
        help='Base directory where train.jsonl/parquet, validation.jsonl/parquet, test.jsonl/parquet files will be created.'
    )
    parser.add_argument(
        '--ratios',
        type=str,
        default='90:5:5',
        help='Train:Validation:Test split ratios (e.g., "90:5:5").'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling.'
    )

    args = parser.parse_args()

    try:
        ratios = tuple(map(int, args.ratios.split(':')))
        if len(ratios) != 3 or sum(ratios) != 100:
            raise ValueError("Ratios must be three integers separated by colons and sum to 100 (e.g., 90:5:5).")
        ratios_float = tuple(r / 100.0 for r in ratios)
    except ValueError as e:
        print(f"Error parsing ratios: {e}")
        return


    split_data(args.input_dir, args.output_base_dir, ratios_float, args.seed)

if __name__ == "__main__":
    main()