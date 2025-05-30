# python pi4pc/data_spliter.py --input_dir data/pi4pc_data --output_base_dir data/pi4pc_splited --ratios 90:5:5

import os
import re
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
from math import floor
import pandas as pd
from datetime import datetime 
# import traceback
import tiktoken

# Calculate the estimated number of tokens in the prompt using tiktoken.
# The size of the prompt passed to the LLM (e.g., GPT) can become very large,
# especially because the patent specification body included in the 'prior_art_specification' field
# can be significantly long. To account for the model's context window limits and API costs,
# data exceeding a certain number of tokens (MAX_PROMPT_TOKENS) is excluded from processing.
# This is to proactively prevent issues that can arise from very long specifications.
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("Warning: 'tiktoken' library not found. Run 'pip install tiktoken' for token-based filtering.")
    print("Skipping token length check. All files will be included regardless of estimated prompt size.")
    TIKTOKEN_AVAILABLE = False
    tokenizer = None

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "testing_prompt_citedParagraph.txt"
MAX_PROMPT_TOKENS = 30000
VERBOSE_LOGGING = False

def _estimate_prompt_tokens(benchmark_data: Dict[str, Any]) -> int:
    """Estimates the token count for the prompt generated from benchmark_data."""
    if not TIKTOKEN_AVAILABLE or tokenizer is None:
        return 0 

    if not PROMPT_TEMPLATE_PATH.is_file():
        print(f"Warning: Prompt template file not found at: {PROMPT_TEMPLATE_PATH}. Cannot estimate tokens.")
        return 0

    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    context_raw = benchmark_data.get("context", {})
    prior_art_raw = benchmark_data.get("prior_art_specification", {})
    options_raw = benchmark_data.get("options", {})

    try: context = json.loads(context_raw) if isinstance(context_raw, str) else context_raw
    except json.JSONDecodeError: context = {}
    try:
        prior_art = json.loads(prior_art_raw) if isinstance(prior_art_raw, str) else prior_art_raw
        if not isinstance(prior_art, dict): prior_art = {}
    except json.JSONDecodeError: prior_art = {}
    try:
        options = json.loads(options_raw) if isinstance(options_raw, str) else options_raw
        if not isinstance(options, dict): options = {}
    except json.JSONDecodeError: options = {}

    claim_num = benchmark_data.get("claim_number", "N/A")
    app_num = benchmark_data.get("application_number", "N/A")

    target_claim_text = "N/A"
    if 'claims' in context and isinstance(context['claims'], list):
         target_claim_text = next((c.get('claim_text', 'N/A')
                                   for c in context['claims']
                                   if isinstance(c, dict) and c.get('claimNumber') == claim_num),
                                  'Claim text not found')

    options_text_list = []
    try:
        sorted_options = sorted(options.items(), key=lambda item: int(item[0]))
    except (ValueError, TypeError):
        sorted_options = options.items()
    for key, text in sorted_options:
        options_text_list.append(f"{key}: {text}")
    options_text = "\n".join(options_text_list)

    format_data = {
        "claim_num": claim_num,
        "app_num": app_num,
        "target_title": context.get("title", "N/A"),
        "target_abstract": context.get("abstract", "N/A"),
        "target_claim_text": target_claim_text,
        "prior_art_patent_id": prior_art.get("patent_id", "N/A"),
        "prior_art_title": prior_art.get("title", "N/A"),
        "prior_art_abstract": prior_art.get("abstract", "N/A"),
        "prior_art_spec_text": prior_art.get("specification", "Full specification text not available."),
        "options_text": options_text
    }

    try:
        filled_prompt = prompt_template.format(**format_data)
        token_count = len(tokenizer.encode(filled_prompt))
        return token_count
    except KeyError as e:
        print(f"Warning: Missing key for token estimation ({benchmark_data.get('application_number')}, cl {benchmark_data.get('claim_number')}): {e}. Returning 0.")
        return 0
    except Exception as e_tok:
        print(f"Warning: Error during token estimation ({benchmark_data.get('application_number')}, cl {benchmark_data.get('claim_number')}): {e_tok}. Returning 0.")
        return 0


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
        jsonl_path = None 

    # Save to Parquet
    try:
        df = pd.DataFrame(data_list)
        for col in ['context', 'prior_art_specification', 'options']:
            if col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    try:
                        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
                    except Exception as e_serial:
                        print(f"Warning: Failed to serialize column '{col}' for Parquet: {e_serial}")

        df.to_parquet(parquet_path, index=False)
        print(f"Successfully saved {split_name} data to {parquet_path}")
    except Exception as e: # Catch pandas/pyarrow errors
        print(f"Error saving {split_name} data to {parquet_path}: {e}")
        print("Make sure 'pandas' and 'pyarrow' are installed ('pip install pandas pyarrow').")
        parquet_path = None

    return jsonl_path, parquet_path

def split_data(input_dir: str, output_base_dir: str, ratios: tuple = (0.9, 0.05, 0.05), seed: int = 42):
    """
    Splits mcq_p_*.json files into train, validation, and test sets based on the target application number.
    Filters out instances where the estimated prompt token count exceeds MAX_PROMPT_TOKENS.
    Saves aggregated data as jsonl and parquet files.

    Args:
        input_dir: Directory containing the source mcq_p_*.json files.
        output_base_dir: Base directory to create train/validation/test output files.
        ratios: Tuple representing the split ratios for train, validation, test.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    skip_log_path = output_base_path / "skipped_long_prompts.log"
    try:
        with open(skip_log_path, 'w', encoding='utf-8') as f_log:
            f_log.write(f"--- Prompt Token Skip Log ({datetime.now().isoformat()}) ---\n")
            f_log.write(f"Max Token Limit: {MAX_PROMPT_TOKENS}\n")
            f_log.write(f"Tokenizer Used (approx): {'tiktoken cl100k_base' if TIKTOKEN_AVAILABLE else 'None (Check Disabled)'}\n\n")
        print(f"Skip log initialized: {skip_log_path}")
    except IOError as e:
        print(f"Error initializing skip log file {skip_log_path}: {e}")

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        return

    mcq_files = list(input_path.glob("mcq_p_*.json"))
    if not mcq_files:
        print(f"Error: No mcq_p_*.json files found in {input_path}")
        return

    print(f"Found {len(mcq_files)} MCQ files.")


    patents = defaultdict(list)
    # Regex to capture the target application number (YYYYYYYY) from mcq_p_pC_rXXXXX_YYYYYYYY_...json
    app_num_pattern = re.compile(r"mcq_p_pC_r\d+_(\d+)_.*\.json")
    skipped_format_count = 0
    for file_path in mcq_files:
        match = app_num_pattern.match(file_path.name)
        if match:
            app_num = match.group(1) # Group 1 captures the target application number
            patents[app_num].append(file_path)
        else:
            # print(f"Warning: Skipping file with unexpected name format: {file_path.name}")
            skipped_format_count += 1
    if skipped_format_count > 0:
         print(f"Warning: Skipped {skipped_format_count} files with unexpected name format.")

    if not patents:
        print("Error: Could not extract target application numbers from any files.")
        return

    unique_app_nums = list(patents.keys())
    random.shuffle(unique_app_nums)
    total_patents = len(unique_app_nums)
    print(f"Found {total_patents} unique target application numbers for splitting.")

    train_size = floor(total_patents * ratios[0])
    valid_size = floor(total_patents * ratios[1])
    test_size = total_patents - train_size - valid_size

    if train_size + valid_size + test_size != total_patents:
        print(f"Info: Split sizes adjusted slightly due to rounding.")

    print(f"Splitting application numbers: Train={train_size}, Validation={valid_size}, Test={test_size}")

    train_apps = unique_app_nums[:train_size]
    valid_apps = unique_app_nums[train_size : train_size + valid_size]
    test_apps = unique_app_nums[train_size + valid_size :]

    train_data, valid_data, test_data = [], [], []
    file_counts = {"train": 0, "validation": 0, "test": 0}
    skipped_token_count = {"train": 0, "validation": 0, "test": 0}
    total_files_processed = 0

    print("\nProcessing, filtering, and aggregating data...")

    print("\n--- Processing Train Data ---")
    for app_num in train_apps:
        for file_path in patents[app_num]:
            total_files_processed += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        estimated_tokens = _estimate_prompt_tokens(content)
                        if VERBOSE_LOGGING:
                            print(f"  [Train] Processing {file_path.name}: Estimated Tokens = {estimated_tokens}")

                        if TIKTOKEN_AVAILABLE and estimated_tokens > MAX_PROMPT_TOKENS:
                            token_skips = 1
                            skipped_token_count["train"] += 1
                            if VERBOSE_LOGGING:
                                print(f"    -> Skipping (Token Limit Exceeded: {estimated_tokens} > {MAX_PROMPT_TOKENS})")
                            try:
                                with open(skip_log_path, 'a', encoding='utf-8') as f_log:
                                    f_log.write(f"Skipped (train): {file_path.name} (Estimated Tokens: {estimated_tokens})\n")
                            except IOError: pass
                            continue

                        train_data.append(content)
                    else:
                         print(f"Warning: Skipping file with non-dict content: {file_path}")
                    file_counts["train"] += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
            if total_files_processed % 100 == 0:
                print(f"... processed {total_files_processed} files ...")

    print("\n--- Processing Validation Data ---")
    for app_num in valid_apps:
        for file_path in patents[app_num]:
            total_files_processed += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        estimated_tokens = _estimate_prompt_tokens(content)
                        if VERBOSE_LOGGING:
                            print(f"  [Validation] Processing {file_path.name}: Estimated Tokens = {estimated_tokens}")

                        if TIKTOKEN_AVAILABLE and estimated_tokens > MAX_PROMPT_TOKENS:
                            token_skips = 1
                            skipped_token_count["validation"] += 1
                            if VERBOSE_LOGGING:
                                print(f"    -> Skipping (Token Limit Exceeded: {estimated_tokens} > {MAX_PROMPT_TOKENS})")
                            try:
                                with open(skip_log_path, 'a', encoding='utf-8') as f_log:
                                    f_log.write(f"Skipped (validation): {file_path.name} (Estimated Tokens: {estimated_tokens})\n")
                            except IOError: pass
                            continue

                        valid_data.append(content)
                    else:
                         print(f"Warning: Skipping file with non-dict content: {file_path}")
                    file_counts["validation"] += 1
            except json.JSONDecodeError:
                 print(f"Warning: Skipping invalid JSON file: {file_path}")
            except Exception as e:
                 print(f"Error reading file {file_path}: {e}")
            if total_files_processed % 100 == 0:
                print(f"... processed {total_files_processed} files ...")

    print("\n--- Processing Test Data ---")
    for app_num in test_apps:
        for file_path in patents[app_num]:
            total_files_processed += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        estimated_tokens = _estimate_prompt_tokens(content)
                        if VERBOSE_LOGGING:
                            print(f"  [Test] Processing {file_path.name}: Estimated Tokens = {estimated_tokens}")

                        if TIKTOKEN_AVAILABLE and estimated_tokens > MAX_PROMPT_TOKENS:
                            token_skips = 1
                            skipped_token_count["test"] += 1
                            if VERBOSE_LOGGING:
                                print(f"    -> Skipping (Token Limit Exceeded: {estimated_tokens} > {MAX_PROMPT_TOKENS})")
                            try:
                                with open(skip_log_path, 'a', encoding='utf-8') as f_log:
                                    f_log.write(f"Skipped (test): {file_path.name} (Estimated Tokens: {estimated_tokens})\n")
                            except IOError: pass
                            continue

                        test_data.append(content)
                    else:
                         print(f"Warning: Skipping file with non-dict content: {file_path}")
                    file_counts["test"] += 1
            except json.JSONDecodeError:
                 print(f"Warning: Skipping invalid JSON file: {file_path}")
            except Exception as e:
                 print(f"Error reading file {file_path}: {e}")
            if total_files_processed % 100 == 0:
                print(f"... processed {total_files_processed} files ...")

    print("\nData aggregation and filtering complete. Saving files...")

    train_jsonl, train_parquet = _save_split_data(train_data, output_base_path, "train")
    valid_jsonl, valid_parquet = _save_split_data(valid_data, output_base_path, "validation")
    test_jsonl, test_parquet = _save_split_data(test_data, output_base_path, "test")

    print("\n--- Splitting Summary ---")
    print(f"Input directory: {input_path}")
    print(f"Output base directory: {output_base_path}")
    print(f"Token Limit for Prompt: {MAX_PROMPT_TOKENS if TIKTOKEN_AVAILABLE else 'N/A (check disabled)'}")
    print("-" * 20)
    print(f"Train set: {len(train_apps)} applications, {file_counts['train']} files processed")
    if TIKTOKEN_AVAILABLE: print(f"  -> Skipped due to token limit: {skipped_token_count['train']}")
    print(f"  -> Records included: {len(train_data)}")
    if train_jsonl: print(f"  -> Saved: {train_jsonl.name}")
    if train_parquet: print(f"  -> Saved: {train_parquet.name}")
    print("-" * 20)
    print(f"Validation set: {len(valid_apps)} applications, {file_counts['validation']} files processed")
    if TIKTOKEN_AVAILABLE: print(f"  -> Skipped due to token limit: {skipped_token_count['validation']}")
    print(f"  -> Records included: {len(valid_data)}")
    if valid_jsonl: print(f"  -> Saved: {valid_jsonl.name}")
    if valid_parquet: print(f"  -> Saved: {valid_parquet.name}")
    print("-" * 20)
    print(f"Test set: {len(test_apps)} applications, {file_counts['test']} files processed")
    if TIKTOKEN_AVAILABLE: print(f"  -> Skipped due to token limit: {skipped_token_count['test']}")
    print(f"  -> Records included: {len(test_data)}")
    if test_jsonl: print(f"  -> Saved: {test_jsonl.name}")
    if test_parquet: print(f"  -> Saved: {test_parquet.name}")
    print("------------------------")
    total_skipped_tokens = sum(skipped_token_count.values())
    if TIKTOKEN_AVAILABLE and total_skipped_tokens > 0:
         print(f"Total items skipped due to token limit: {total_skipped_tokens}. See log: {skip_log_path}")

def main_cli():
    parser = argparse.ArgumentParser(description="Split mcq_p_*.json benchmark files into train/validation/test sets based on target application number, filter by estimated prompt token length, saving aggregated data as jsonl and parquet.")
    parser.add_argument('--input_dir', type=str, default='data/mcq_cited_paragraph', help='Directory containing the source mcq_p_*.json files.')
    parser.add_argument('--output_base_dir', type=str, default='data/mcq_cited_paragraph_splited', help='Base directory where train/validation/test files will be created.')
    parser.add_argument('--ratios', type=str, default='90:5:5', help='Train:Validation:Test split ratios (e.g., "90:5:5").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling.')
    parser.add_argument('--max_tokens', type=int, default=30000, help='Maximum estimated prompt tokens allowed. Set 0 to disable.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging including token counts for each file.')

    args = parser.parse_args()

    global VERBOSE_LOGGING
    VERBOSE_LOGGING = args.verbose

    global MAX_PROMPT_TOKENS
    if args.max_tokens > 0:
        MAX_PROMPT_TOKENS = args.max_tokens
    elif args.max_tokens == 0:
         print("Token limit check disabled by --max_tokens 0.")
         global TIKTOKEN_AVAILABLE
         TIKTOKEN_AVAILABLE = False
         MAX_PROMPT_TOKENS = float('inf')

    try:
        ratios = tuple(map(int, args.ratios.split(':')))
        if len(ratios) != 3 or sum(ratios) != 100:
            raise ValueError("Ratios must be three integers separated by colons and sum to 100 (e.g., 90:5:5).")
        ratios_float = tuple(r / 100.0 for r in ratios)
    except ValueError as e:
        print(f"Error parsing ratios: {e}")
        return

    try:
        import pandas
        import pyarrow
    except ImportError:
        print("Error: 'pandas' and 'pyarrow' libraries are required.")
        print("Please install them using: pip install pandas pyarrow")
        return

    split_data(args.input_dir, args.output_base_dir, ratios_float, args.seed)

if __name__ == "__main__":
    main_cli()