# python benchmarks/noc4pc/data_spliter.py --input_dir  --output_base_dir data/record_splited --ratios 90:5:5

import os
import re
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from math import floor
import pandas as pd
import sys # For error exit

def _save_split_data(data_list: list, output_path_base: Path, split_name: str):
    if not data_list:
        print(f"Warning: No data found for split '{split_name}'. Skipping file generation.")
        return None, None

    jsonl_path = output_path_base / f"{split_name}.jsonl"
    parquet_path = output_path_base / f"{split_name}.parquet"

    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f_jsonl:
            for item in data_list:
                for key in ['context', 'prior_art_specifications', 'answer']:
                     if key in item and isinstance(item[key], (dict, list)):
                          item[key] = json.dumps(item[key], ensure_ascii=False)
                f_jsonl.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {split_name} data to {jsonl_path} ({len(data_list)} records)")
    except IOError as e:
        print(f"Error saving {split_name} data to {jsonl_path}: {e}")
        jsonl_path = None
    except Exception as e_jsonl_dump:
         print(f"Error processing data during JSONL dump for {split_name}: {e_jsonl_dump}")
         jsonl_path = None

    try:
        df = pd.DataFrame(data_list)
        for col in ['context', 'prior_art_specifications', 'answer']:
             if col in df.columns:
                 mask = df[col].apply(lambda x: isinstance(x, (dict, list)))
                 if mask.any():
                      df.loc[mask, col] = df.loc[mask, col].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df.to_parquet(parquet_path, index=False)
        print(f"Successfully saved {split_name} data to {parquet_path} ({len(df)} records)")
    except Exception as e_parquet:
        print(f"Error saving {split_name} data to {parquet_path}: {e_parquet}")
        print("Make sure 'pandas' and 'pyarrow' are installed ('pip install pandas pyarrow').")
        parquet_path = None

    return jsonl_path, parquet_path

def split_data(input_dir: str, output_base_dir: str, ratios: tuple = (0.9, 0.05, 0.05), seed: int = 42):
    random.seed(seed)
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        return

    rejection_files = list(input_path.glob("rejection_*.json"))
    if not rejection_files:
        print(f"Error: No rejection_*.json files found in {input_path}")
        return

    print(f"Found {len(rejection_files)} rejection benchmark files.")

    patents = defaultdict(list)
    patent_num_pattern = re.compile(r"rejection_r\d+_(\d+)_cl\d+\.json")

    files_skipped_format = 0
    for file_path in rejection_files:
        match = patent_num_pattern.match(file_path.name)
        if match:
            patent_num = match.group(1)
            patents[patent_num].append(file_path)
        else:
            print(f"Warning: Skipping file with unexpected name format: {file_path.name}")
            files_skipped_format += 1

    if not patents:
        print("Error: Could not extract target patent numbers from any files.")
        return

    if files_skipped_format > 0:
        print(f"Warning: Skipped {files_skipped_format} files due to unexpected naming format.")

    unique_patent_nums = list(patents.keys())
    random.shuffle(unique_patent_nums)
    total_patents = len(unique_patent_nums)
    print(f"Found {total_patents} unique target patent numbers for splitting.")

    train_size = floor(total_patents * ratios[0])
    valid_size = floor(total_patents * ratios[1])
    test_size = total_patents - train_size - valid_size

    print(f"Splitting patent numbers: Train={train_size}, Validation={valid_size}, Test={test_size}")

    train_patents = unique_patent_nums[:train_size]
    valid_patents = unique_patent_nums[train_size : train_size + valid_size]
    test_patents = unique_patent_nums[train_size + valid_size :]

    train_data, valid_data, test_data = [], [], []
    file_counts = {"train": 0, "validation": 0, "test": 0}
    error_counts = {"train": 0, "validation": 0, "test": 0}

    print("\nProcessing and aggregating data...")

    def process_patent_set(patent_num_list, data_list, error_counter):
        count = 0
        for patent_num in patent_num_list:
            for file_path in patents[patent_num]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, dict):
                            data_list.append(content)
                            count += 1
                        else:
                            print(f"Warning: Skipping file with non-dict content: {file_path}")
                            error_counter += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON file: {file_path}")
                    error_counter += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    error_counter += 1
        return count, error_counter

    file_counts["train"], error_counts["train"] = process_patent_set(train_patents, train_data, error_counts["train"])
    file_counts["validation"], error_counts["validation"] = process_patent_set(valid_patents, valid_data, error_counts["validation"])
    file_counts["test"], error_counts["test"] = process_patent_set(test_patents, test_data, error_counts["test"])

    total_files_processed = sum(file_counts.values())
    total_errors = sum(error_counts.values())
    print(f"Data aggregation complete. Processed {total_files_processed} files with {total_errors} errors/skips during reading.")

    print("Saving aggregated split files...")
    output_base_path.mkdir(parents=True, exist_ok=True)
    train_jsonl, train_parquet = _save_split_data(train_data, output_base_path, "train")
    valid_jsonl, valid_parquet = _save_split_data(valid_data, output_base_path, "validation")
    test_jsonl, test_parquet = _save_split_data(test_data, output_base_path, "test")

    print("\n--- Splitting Summary ---")
    print(f"Input directory: {input_path}")
    print(f"Output base directory: {output_base_path}")
    print("-" * 20)
    print(f"Train set: {len(train_patents)} patents, {file_counts['train']} files processed, {len(train_data)} records aggregated")
    if train_jsonl: print(f"  -> {train_jsonl.name}")
    if train_parquet: print(f"  -> {train_parquet.name}")
    if error_counts['train'] > 0: print(f"  -> Errors/skips reading train files: {error_counts['train']}")
    print("-" * 20)
    print(f"Validation set: {len(valid_patents)} patents, {file_counts['validation']} files processed, {len(valid_data)} records aggregated")
    if valid_jsonl: print(f"  -> {valid_jsonl.name}")
    if valid_parquet: print(f"  -> {valid_parquet.name}")
    if error_counts['validation'] > 0: print(f"  -> Errors/skips reading validation files: {error_counts['validation']}")
    print("-" * 20)
    print(f"Test set: {len(test_patents)} patents, {file_counts['test']} files processed, {len(test_data)} records aggregated")
    if test_jsonl: print(f"  -> {test_jsonl.name}")
    if test_parquet: print(f"  -> {test_parquet.name}")
    if error_counts['test'] > 0: print(f"  -> Errors/skips reading test files: {error_counts['test']}")
    print("------------------------")

def main():
    parser = argparse.ArgumentParser(description="Split rejection_*.json benchmark files into train/validation/test sets based on target patent number, saving aggregated data as jsonl and parquet.")
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/rejection',
        help='Directory containing the source rejection_*.json files.'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='data/rejection_splited',
        help='Base directory where train.jsonl/parquet, validation.jsonl/parquet, test.jsonl/parquet files will be created.'
    )
    parser.add_argument(
        '--ratios',
        type=str,
        default='90:5:5',
        help='Train:Validation:Test split ratios (e.g., "90:5:5"). Must sum to 100.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling.'
    )

    args = parser.parse_args()

    try:
        ratios_int = tuple(map(int, args.ratios.split(':')))
        if len(ratios_int) != 3 or sum(ratios_int) != 100:
            raise ValueError("Ratios must be three integers separated by colons and sum to 100 (e.g., 90:5:5).")
        ratios_float = tuple(r / 100.0 for r in ratios_int)
    except ValueError as e:
        print(f"Error parsing ratios: {e}")
        return

    try:
        import pandas
        import pyarrow
    except ImportError:
        print("FATAL ERROR: 'pandas' and 'pyarrow' libraries are required for Parquet output.", file=sys.stderr)
        print("Please install them using: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    split_data(args.input_dir, args.output_base_dir, ratios_float, args.seed)

if __name__ == "__main__":
    main()