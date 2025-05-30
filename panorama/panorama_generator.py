#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python -m pip install tqdm pandas pyarrow

import os
import json
import glob
import re
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('panorama_generator.log')
    ]
)
logger = logging.getLogger('panorama_generator')

def setup_directories():
    """Creates the necessary directory structure."""
    os.makedirs('data/panorama', exist_ok=True)
    os.makedirs('data/panorama/parquet', exist_ok=True)
    logger.info("Directory structure verified.")

def extract_info_from_filename(filename):
    """
    Extracts ID and patent application number from the filename.
    
    Args:
        filename (str): Filename (e.g., pC_r00002_14937767.json)
    
    Returns:
        tuple: (ID, patent application number)
    """
    try:
        match = re.search(r'_r(\d+)_(\d+)\.json$', filename)
        if match:
            return match.group(1), match.group(2)
        else:
            logger.error(f"Filename format error: {filename}")
            return None, None
    except Exception as e:
        logger.error(f"Error processing filename: {filename}, Error: {str(e)}")
        return None, None

def merge_files(pc_file_path, rec_file_path, output_path):
    """
    Merges the contents of pC and rec files to create a panorama file.
    
    Args:
        pc_file_path (str): Path to the pC file.
        rec_file_path (str): Path to the rec file.
        output_path (str): Path to the output file.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(pc_file_path, 'r', encoding='utf-8') as f:
            pc_data = json.load(f)
        
        with open(rec_file_path, 'r', encoding='utf-8') as f:
            rec_data = json.load(f)
        
        panorama_data = {**rec_data, "parsed_CTNF": pc_data.get("claims", [])}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(panorama_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error merging files: PC={pc_file_path}, REC={rec_file_path}, Error: {str(e)}")
        return False

def create_jsonl_and_parquet():
    """
    Converts all panorama JSON files to JSONL and Parquet formats.
    
    Returns:
        tuple: (Number of successful JSONL entries, number of successful Parquet entries, number of conversion failures)
    """
    try:
        logger.info("Starting JSONL and Parquet file generation.")
        
        panorama_files = glob.glob('data/panorama/panorama_*.json')
        
        if not panorama_files:
            logger.warning("No panorama files to convert.")
            return 0, 0, 0

        jsonl_path = 'data/panorama/parquet/panorama_data.jsonl'
        parquet_path = 'data/panorama/parquet/panorama_data.parquet'

        jsonl_success_count = 0
        parquet_success_count = 0
        conversion_error_count = 0

        with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            for file_path in tqdm(panorama_files, desc="Generating JSONL"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                    jsonl_success_count += 1
                except Exception as e:
                    logger.error(f"Error generating JSONL (file: {file_path}): {str(e)}")
                    conversion_error_count += 1
        
        logger.info(f"JSONL file generation complete: {jsonl_path}")

        records = []
        for file_path in tqdm(panorama_files, desc="Loading data for Parquet conversion"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                records.append(data)
            except Exception as e:
                logger.error(f"Error loading data for Parquet conversion (file: {file_path}): {str(e)}")
                conversion_error_count += 1
        
        if records:
            df = pd.DataFrame(records)
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Parquet file generation complete: {parquet_path}")
            parquet_success_count = len(records)
        else:
            logger.warning("No data for Parquet conversion.")
            parquet_success_count = 0
        
        return jsonl_success_count, parquet_success_count, conversion_error_count
    
    except Exception as e:
        logger.error(f"Error generating JSONL and Parquet files: {str(e)}")
        return 0, 0, 0

def process_files():
    """
    Processes all files to generate panorama files.
    
    Returns:
        dict: Dictionary containing processing statistics.
    """
    pc_files = glob.glob('data/parsed_CTNF/pC_*.json')
    total_pc_files = len(pc_files)
    logger.info(f"Number of pC files found: {total_pc_files}")
    
    successful_count = 0
    error_count = 0
    not_found_count = 0
    not_found_files = []
    
    for pc_file_path in tqdm(pc_files, desc="Processing files"):
        pc_filename = os.path.basename(pc_file_path)
        id_number, appl_number = extract_info_from_filename(pc_filename)
        
        if id_number is None or appl_number is None:
            logger.error(f"Failed to extract file information: {pc_filename}")
            error_count += 1
            continue

        rec_filename = f"rec_r{id_number}_{appl_number}.json"
        rec_file_path = os.path.join('data/record', rec_filename)

        if not os.path.exists(rec_file_path):
            logger.warning(f"Corresponding rec file not found: {rec_filename}")
            not_found_count += 1
            not_found_files.append(pc_filename)
            continue

        panorama_filename = f"panorama_r{id_number}_{appl_number}.json"
        panorama_file_path = os.path.join('data/panorama', panorama_filename)

        if merge_files(pc_file_path, rec_file_path, panorama_file_path):
            logger.debug(f"File merge successful: {panorama_filename}")
            successful_count += 1
        else:
            error_count += 1

    if not_found_files:
        not_found_list_path = 'data/panorama/unmatched_files.txt'
        try:
            with open(not_found_list_path, 'w', encoding='utf-8') as f:
                for filename in not_found_files:
                    f.write(f"{filename}\n")
            logger.info(f"List of unmatched files saved: {not_found_list_path}")
        except Exception as e:
            logger.error(f"Error saving list of unmatched files: {str(e)}")
    
    stats = {
        "total_pc_files": total_pc_files,
        "successful": successful_count,
        "rec_not_found": not_found_count,
        "error": error_count
    }
    
    logger.info(f"Processing complete: Successful={successful_count}, rec files missing={not_found_count}, Errors={error_count}")
    return stats

def print_summary(file_stats, conversion_stats=None):
    """
    Displays processing statistics.
    
    Args:
        file_stats (dict): File processing statistics.
        conversion_stats (tuple, optional): Conversion statistics (JSONL success, Parquet success, conversion failures).
    """
    total = file_stats["total_pc_files"]
    success = file_stats["successful"]
    not_found = file_stats["rec_not_found"]
    error = file_stats["error"]
    
    success_rate = (success / total) * 100 if total > 0 else 0
    
    print("\n" + "="*60)
    print("                     Processing Result Summary                     ")
    print("="*60)
    print(f"Total pC files: {total}")
    print(f"Successfully processed files: {success} ({success_rate:.1f}%)")
    print(f"Cases where matching rec file was not found: {not_found}")
    print(f"Errors during processing: {error}")
    
    if conversion_stats:
        jsonl_success, parquet_success, conversion_error = conversion_stats
        print("\n" + "-"*60)
        print("               Conversion Statistics Summary                ")
        print("-"*60)
        print(f"JSONL conversion successful: {jsonl_success}")
        print(f"Parquet conversion successful: {parquet_success}")
        print(f"Errors during conversion: {conversion_error}")
    
    print("="*60 + "\n")

def main():
    logger.info("Panorama Generator started.")
    
    try:
        setup_directories()
        file_stats = process_files()
        conversion_stats = None
        
        if file_stats["successful"] > 0:
            conversion_stats = create_jsonl_and_parquet()
        else:
            logger.warning("No panorama files generated, so JSONL and Parquet will not be created.")
        
        print_summary(file_stats, conversion_stats)
        
        logger.info("Panorama Generator finished.")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
