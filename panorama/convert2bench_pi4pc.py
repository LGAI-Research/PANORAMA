import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import traceback
from datetime import datetime
import csv
import sys
import pandas as pd
import random
from collections import defaultdict

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"JSON parsing error: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"File reading error {filepath}: {e}", file=sys.stderr)
        return None

def extract_core_patent_number(raw_num: str) -> str | None:
    if not isinstance(raw_num, str): return None
    cleaned = re.sub(r'^(US|EP|WO|JP|KR)\s*|\s|,|-', '', raw_num.upper())
    match = re.search(r'([\dX]+)$', cleaned);
    if match: return match.group(1)
    if re.fullmatch(r'[\dX]+', cleaned): return cleaned
    digits = re.findall(r'[\dX]', cleaned);
    if digits: return "".join(digits)
    return None

def find_spec_file(spec_dir: Path, cited_patent_num_cleaned: str) -> Path | None:
    if not cited_patent_num_cleaned: return None
    target_filename_pattern = f"spec_txt_{cited_patent_num_cleaned}_parsed.json"
    potential_file = spec_dir / target_filename_pattern
    if potential_file.is_file(): return potential_file
    return None

def parse_paragraph_key(raw_key: Any) -> int | None:
    if isinstance(raw_key, int): return raw_key
    if not isinstance(raw_key, str): return None
    try: return int(raw_key)
    except ValueError:
        if re.fullmatch(r'0+\d+', raw_key):
            try: return int(raw_key)
            except ValueError: pass
        parts = re.split(r'[-–—−]', raw_key)
        if len(parts) > 1:
            last_part = parts[-1].strip()
            numeric_match = re.search(r'\d+$', last_part)
            if numeric_match:
                try: return int(numeric_match.group(0))
                except ValueError: pass
            numeric_match_in_last = re.search(r'\d+', last_part)
            if numeric_match_in_last:
                try: return int(numeric_match_in_last.group(0))
                except ValueError: pass
    return None

def get_paragraph_content(spec_data: Dict[str, Any], target_key: int) -> str | None:
    if not spec_data or "items" not in spec_data:
        return None
    for item in spec_data["items"]:
        if isinstance(item, dict):
            parsed_key = parse_paragraph_key(item.get("key"))
            if parsed_key == target_key:
                return item.get("content")
    return None

def get_random_paragraph_from_spec(spec_data: Dict[str, Any], exclude_keys: Set[int] = set()) -> Tuple[int, str] | None:
    if not spec_data or "items" not in spec_data:
        return None
    
    eligible_paragraphs = []
    for item in spec_data["items"]:
        if isinstance(item, dict):
            key = parse_paragraph_key(item.get("key"))
            content = item.get("content")
            if key is not None and content and key not in exclude_keys:
                eligible_paragraphs.append((key, content))

    if not eligible_paragraphs:
        return None
    
    return random.choice(eligible_paragraphs)


def parse_date(date_string):
    if not date_string or not isinstance(date_string, str): return None
    try: date_string_cleaned = date_string.split('.')[0]; return datetime.fromisoformat(date_string_cleaned.replace('Z', '+00:00'))
    except ValueError: pass
    try: return datetime.strptime(date_string.split('T')[0], '%Y-%m-%d')
    except ValueError: pass
    return None

def get_main_class(record_data):
    patent_class_full = record_data.get("class", "");
    if patent_class_full and isinstance(patent_class_full, str): return patent_class_full.split('/')[0]
    return None

def find_negative_patent_id_pool(current_index, target_class, current_filing_date,
                                 valid_records_info, record_dir, exclude_ids, target_pool_size=50):
    negative_pool = set()
    processed_app_ids = set()
    
    current_filing_dt = parse_date(current_filing_date)
    if not current_filing_dt: return []

    current_app_num = valid_records_info[current_index]['app_num']
    
    search_indices = list(range(current_index + 1, len(valid_records_info))) + list(range(0, current_index))

    for i in search_indices:
        if len(negative_pool) >= target_pool_size: break

        next_record_info = valid_records_info[i]
        next_rec_num = next_record_info["rec_num"]; next_app_num = next_record_info["app_num"]

        if next_app_num == current_app_num or next_app_num in processed_app_ids: continue
        processed_app_ids.add(next_app_num)

        next_record_file = next(record_dir.glob(f"rec_r{next_rec_num:05d}_{next_app_num}.json"), None)
        if not next_record_file: continue
        next_record_data = load_json(next_record_file)
        if not next_record_data: continue

        next_class = get_main_class(next_record_data)
        if not next_class or next_class == target_class: continue

        next_filing_dt = parse_date(next_record_data.get("filingDate"))
        if not next_filing_dt or not (next_filing_dt < current_filing_dt): continue

        
        cited_by_examiner = next_record_data.get("patentsCitedByExaminer", [])
        if isinstance(cited_by_examiner, list):
            for cited in cited_by_examiner:
                if isinstance(cited, dict):
                    raw_id = cited.get("referenceIdentifier")
                    cleaned_id = extract_core_patent_number(str(raw_id))
                    if cleaned_id and cleaned_id not in exclude_ids:
                        negative_pool.add(cleaned_id)
                        if len(negative_pool) >= target_pool_size: break
            if len(negative_pool) >= target_pool_size: break
            
    return list(negative_pool)


def log_status(writer, file_handle, timestamp, pc_file, claim_num, cited_raw, status, details):
    entry = {
        'Timestamp': timestamp, 'pC_File': pc_file if pc_file else 'N/A',
        'Claim_Number': claim_num if claim_num else 'N/A',
        'Cited_Patent_Raw': cited_raw if cited_raw else 'N/A',
        'Status': status, 'Details': str(details)[:2000]
    }
    try:
        if writer and file_handle: writer.writerow(entry); file_handle.flush(); return 1
        else: print(f"ERROR: Log writer not initialized. Entry lost: {entry}", file=sys.stderr); return 0
    except Exception as e: print(f"Error writing log entry: {e} - Entry: {entry}", file=sys.stderr); return 0

def find_cited_details_in_record(record_data: Dict[str, Any], cleaned_cited_num_to_find: str) -> Dict[str, Any]:
    default_details = {"title": "N/A", "abstract": "N/A", "claims": []}
    if not record_data or not cleaned_cited_num_to_find:
        return default_details

    examiner_citations = record_data.get("patentsCitedByExaminer", [])
    if not isinstance(examiner_citations, list):
        return default_details

    for cited_patent in examiner_citations:
        if isinstance(cited_patent, dict):
            ref_id_raw = cited_patent.get("referenceIdentifier")
            if ref_id_raw:
                normalized_ref_id = extract_core_patent_number(str(ref_id_raw))
                if normalized_ref_id and normalized_ref_id == cleaned_cited_num_to_find:
                    title = cited_patent.get("title")
                    abstract = cited_patent.get("abstract")
                    claims = cited_patent.get("claims", [])
                    if not isinstance(claims, list):
                        claims = []

                    return {
                        "title": str(title) if title else "N/A",
                        "abstract": str(abstract) if abstract else "N/A",
                        "claims": claims
                    }
    return default_details

def create_cited_paragraph_pi4pc(paragraph_dir: Path, spec_dir: Path, record_dir: Path, output_dir: Path, error_dir: Path, output_file_jsonl: Path, output_file_parquet: Path):
    print("--- Cited Paragraph PI4PC Benchmark Generation Started (New Logic) ---")
    print(f"Paragraph Source (pC files): {paragraph_dir}")
    print(f"Specification Source: {spec_dir}")
    print(f"Record Source: {record_dir}")
    print(f"PI4PC Output Directory (individual files): {output_dir}")
    print(f"Error/Log Directory: {error_dir}")
    print(f"Aggregated JSONL Output File: {output_file_jsonl}")
    print(f"Aggregated Parquet Output File: {output_file_parquet}")

    try:
        paragraph_dir.mkdir(parents=True, exist_ok=True); spec_dir.mkdir(parents=True, exist_ok=True)
        record_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        if output_file_jsonl: output_file_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if output_file_parquet: output_file_parquet.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e: print(f"FATAL: Could not create directories: {e}", file=sys.stderr); return

    if not paragraph_dir.is_dir(): print(f"FATAL: Paragraph dir not found: {paragraph_dir}", file=sys.stderr); return
    if not spec_dir.is_dir(): print(f"FATAL: Specification dir not found: {spec_dir}", file=sys.stderr); return
    if not record_dir.is_dir(): print(f"FATAL: Record dir not found: {record_dir}", file=sys.stderr); return

    pc_files = sorted(list(paragraph_dir.glob("pC_*.json")))
    if not pc_files: print(f"Error: No pC files found in {paragraph_dir}", file=sys.stderr); return
    total_pc_files = len(pc_files)
    print(f"Found {total_pc_files} pC files to process.")

    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = error_dir / f'cited_paragraph_pi4pc_log_{timestamp_log}.csv'
    log_csv_file = None; log_writer = None
    log_fieldnames = ['Timestamp', 'pC_File', 'Claim_Number', 'Cited_Patent_Raw', 'Status', 'Details']
    try:
        log_csv_file = open(log_path, 'w', newline='', encoding='utf-8-sig')
        log_writer = csv.DictWriter(log_csv_file, fieldnames=log_fieldnames)
        log_writer.writeheader()
        print(f"Log file initialized: {log_path}")
    except IOError as e: print(f"FATAL: Cannot init log file {log_path}: {e}", file=sys.stderr); return

    all_pi4pc_data = []
    pi4pc_generated_count = 0
    files_processed = 0
    files_skipped_load_error = 0
    instances_skipped_missing_data = 0 
    instances_skipped_option_failure = 0 
    
    valid_records_info = []
    print("Gathering record file information...")
    for rec_file in record_dir.glob("rec_*.json"):
        match_rec = re.search(r'rec_r(\d+)_(\d+)\.json', rec_file.name)
        if match_rec:
            try:
                rec_num = int(match_rec.group(1))
                app_num = match_rec.group(2)
                valid_records_info.append({'rec_num': rec_num, 'app_num': app_num})
            except ValueError: pass
    valid_records_info.sort(key=lambda x: x['rec_num'])
    print(f"Found {len(valid_records_info)} records to potentially use for negative sampling.")

    spec_txt_dir = spec_dir.parent
    if not spec_txt_dir.is_dir():
        print(f"FATAL: Specification text directory not found at {spec_txt_dir} (derived from {spec_dir})", file=sys.stderr)
        log_status(log_writer, log_csv_file, datetime.now().isoformat(), None, None, None, "Fatal", f"Spec text dir not found: {spec_txt_dir}")
        if log_csv_file: log_csv_file.close()
        return
    print(f"Specification Text Source: {spec_txt_dir}")

    try:
        for i, pc_file in enumerate(pc_files):
            try: 
                files_processed += 1
                print(f"\n--- Processing file {files_processed}/{total_pc_files}: {pc_file.name} ---")
                
                pc_data = load_json(pc_file)
                if not pc_data:
                    raise RuntimeError("Failed to load or parse pC JSON.")

                match_pc = re.search(r'pC_r\d+_(\d+)(?:_cl\d+)?\.json', pc_file.name)
                if not match_pc:
                    raise RuntimeError("Could not extract target app number from filename.")
                target_app_num = match_pc.group(1)

                record_file = next(record_dir.glob(f"rec_*_{target_app_num}.json"), None)
                if not record_file:
                    raise RuntimeError(f"Record file for app {target_app_num} not found.")
                record_data = load_json(record_file)
                if not record_data:
                    raise RuntimeError(f"Failed to load record file {record_file.name}.")

                target_spec_text = None
                target_spec_txt_path = spec_txt_dir / f"spec_txt_{target_app_num}.txt"
                try:
                    if target_spec_txt_path.is_file():
                        target_spec_text = target_spec_txt_path.read_text(encoding='utf-8')
                    else:
                        log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Warning: Target Spec Text Missing", f"Target spec text file not found: {target_spec_txt_path.name}")
                except Exception as e_spec_txt:
                    log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Error: Reading Target Spec Text", f"Failed to read {target_spec_txt_path.name}: {e_spec_txt}")

                target_context = {
                    "title": record_data.get("title", "N/A"),
                    "abstract": record_data.get("abstract", "N/A"),
                    "claims": record_data.get("initialClaims", [])
                }
                if target_context["title"] == "N/A" or target_context["abstract"] == "N/A" or not target_context["claims"]:
                     log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Skip File: Missing Target Context", f"Essential target context (title, abstract, or claims) missing in record file {record_file.name}.")
                     files_skipped_load_error += 1
                     continue
                all_cited_pairs_in_pc = set()
                claim_citations = defaultdict(lambda: defaultdict(set))
                patent_citations_in_pc = defaultdict(set)

                claims_in_pc = pc_data.get("claims", [])
                if not isinstance(claims_in_pc, list):
                    log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Warning: Invalid claims format", "'claims' field is not a list in pC file. Skipping claims processing for this file.")
                else:
                    for claim_info in claims_in_pc:
                        if not isinstance(claim_info, dict): continue
                        claim_num = claim_info.get("claimNumber")
                        reasons = claim_info.get("reasons", [])
                        if not isinstance(reasons, list): continue
                        
                        for reason in reasons:
                            if not isinstance(reason, dict): continue
                            cited_patents = reason.get("citedPatents", [])
                            if not isinstance(cited_patents, list): continue
                            
                            for cited_patent in cited_patents:
                                if not isinstance(cited_patent, dict): continue
                                raw_cited_num = cited_patent.get("patentNum")
                                text_keys_raw = cited_patent.get("text", [])
                                
                                if raw_cited_num and isinstance(text_keys_raw, list):
                                    cleaned_cited_num = extract_core_patent_number(str(raw_cited_num))
                                    if cleaned_cited_num:
                                        current_claim_patent_keys = set()
                                        for key_raw in text_keys_raw:
                                            parsed_key = parse_paragraph_key(key_raw)
                                            if parsed_key is not None:
                                                all_cited_pairs_in_pc.add((cleaned_cited_num, parsed_key))
                                                patent_citations_in_pc[cleaned_cited_num].add(parsed_key)
                                                current_claim_patent_keys.add(parsed_key)

                                        if claim_num is not None and current_claim_patent_keys:
                                             claim_citations[claim_num][cleaned_cited_num].update(current_claim_patent_keys)

                if not claim_citations:
                     log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Info: No citations", "No valid claim citations found in pC file.")
                     continue

                for claim_num, cited_map in claim_citations.items():
                    for cited_patent_id, gold_keys_for_claim in cited_map.items():
                        if not gold_keys_for_claim: continue

                        try:
                            gold_para_info = None
                            silver_para_info = None
                            negative_para_info_list = []
                            prior_art_details = None
                            prior_art_spec_text = None
                            spec_path = None
                            spec_data = None
                            all_keys_in_spec = set()
                            all_paragraphs_in_spec = {}
                            valid_gold_keys = set()
                            chosen_gold_key = None
                            gold_content = None
                            valid_silver_keys = []
                            chosen_silver_key = None
                            silver_content = None
                            valid_negative_keys = []
                            selected_negative_keys = []
                            options_list = []
                            options_dict = {}
                            gold_answer_keys = []
                            silver_answer_keys = []
                            negative_answer_keys = []


                            prior_art_details = find_cited_details_in_record(record_data, cited_patent_id)
                            prior_art_details["patent_id"] = cited_patent_id

                            if prior_art_details["title"] == "N/A" or prior_art_details["abstract"] == "N/A":
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Missing Prior Art Details", f"Essential details (title/abstract) missing in record for {cited_patent_id}")
                                instances_skipped_missing_data += 1; continue

                            prior_art_spec_txt_path = spec_txt_dir / f"spec_txt_{cited_patent_id}.txt"
                            try:
                                if prior_art_spec_txt_path.is_file():
                                    prior_art_spec_text = prior_art_spec_txt_path.read_text(encoding='utf-8')
                                    if not prior_art_spec_text:
                                        log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Empty Prior Art Spec Text", f"Prior art spec text file is empty: {prior_art_spec_txt_path.name}")
                                        instances_skipped_missing_data += 1; continue
                                else:
                                    log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Missing Prior Art Spec Text", f"Prior art spec text file not found: {prior_art_spec_txt_path.name}")
                                    instances_skipped_missing_data += 1; continue
                            except Exception as e_pa_spec_txt:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Error Reading Prior Art Spec Text", f"Failed to read {prior_art_spec_txt_path.name}: {e_pa_spec_txt}")
                                instances_skipped_missing_data += 1; continue
                            prior_art_details["specification"] = prior_art_spec_text

                            spec_path = find_spec_file(spec_dir, cited_patent_id)
                            if not spec_path:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Parsed Spec Missing", f"Parsed spec JSON not found for {cited_patent_id}")
                                instances_skipped_missing_data += 1; continue
                            spec_data = load_json(spec_path)
                            if not spec_data or "items" not in spec_data or not spec_data["items"]:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Parsed Spec Empty or Invalid", f"Parsed spec JSON invalid or empty for {cited_patent_id} at {spec_path.name}")
                                instances_skipped_missing_data += 1; continue

                            all_keys_in_spec = set()
                            all_paragraphs_in_spec = {}
                            for item in spec_data.get("items", []):
                                if isinstance(item, dict):
                                    key = parse_paragraph_key(item.get("key"))
                                    content = item.get("content")
                                    if key is not None and content:
                                        all_keys_in_spec.add(key)
                                        all_paragraphs_in_spec[key] = content
                            if not all_keys_in_spec:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: No Valid Paragraphs in Spec", f"No valid paragraphs (key, content) found in spec file {spec_path.name}")
                                instances_skipped_missing_data += 1; continue

                            all_cited_keys_for_this_patent = patent_citations_in_pc.get(cited_patent_id, set())

                            potential_silver_keys = all_cited_keys_for_this_patent - gold_keys_for_claim
                            potential_negative_keys = all_keys_in_spec - all_cited_keys_for_this_patent

                            valid_gold_keys = gold_keys_for_claim.intersection(all_keys_in_spec)
                            if not valid_gold_keys:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Gold Key Not Found in Spec", f"Cited gold keys {gold_keys_for_claim} not found or invalid in spec {spec_path.name}. Keys found in spec: {all_keys_in_spec}")
                                instances_skipped_missing_data += 1; continue

                            chosen_gold_key = random.choice(list(valid_gold_keys))
                            gold_content = all_paragraphs_in_spec.get(chosen_gold_key)
                            if not gold_content:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Gold Content Missing", f"Gold key {chosen_gold_key} found, but content is missing in spec data for {spec_path.name}.")
                                instances_skipped_missing_data += 1; continue
                            gold_para_info = {"id": cited_patent_id, "key": chosen_gold_key, "content": gold_content}

                            valid_silver_keys = list(potential_silver_keys.intersection(all_keys_in_spec))
                            chosen_silver_key = None
                            if valid_silver_keys:
                                chosen_silver_key = random.choice(valid_silver_keys)
                                silver_content = all_paragraphs_in_spec.get(chosen_silver_key)
                                if silver_content:
                                    silver_para_info = {"id": cited_patent_id, "key": chosen_silver_key, "content": silver_content}
                                else:
                                    log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Warning: Silver Content Missing", f"Silver key {chosen_silver_key} selected, but content missing in spec data {spec_path.name}. Proceeding without silver.")


                            num_negatives_needed = 4 if silver_para_info is None else 3
                            valid_negative_keys = list(potential_negative_keys)
                            random.shuffle(valid_negative_keys)


                            selected_negative_keys = []
                            for neg_key in valid_negative_keys:
                                neg_content = all_paragraphs_in_spec.get(neg_key)
                                if neg_content:
                                    negative_para_info_list.append({"id": cited_patent_id, "key": neg_key, "content": neg_content})
                                    selected_negative_keys.append(neg_key)
                                    if len(selected_negative_keys) == num_negatives_needed:
                                        break

                            options_list = [gold_para_info] + ([silver_para_info] if silver_para_info else []) + negative_para_info_list

                            if len(options_list) < 5:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Skip Instance: Insufficient Options", f"Could only generate {len(options_list)} valid options (Gold: {1}, Silver: {1 if silver_para_info else 0}, Neg: {len(selected_negative_keys)}). Needed 5.")
                                instances_skipped_option_failure += 1
                                continue

                            random.shuffle(options_list)

                            options_dict = {}
                            gold_answer_keys = []
                            silver_answer_keys = []
                            negative_answer_keys = []

                            for para_info in options_list:
                                key = para_info["key"]
                                content = para_info["content"]
                                options_dict[key] = content

                                if key == chosen_gold_key:
                                    gold_answer_keys.append(key)
                                elif silver_para_info and key == chosen_silver_key:
                                    silver_answer_keys.append(key)
                                elif key in selected_negative_keys:
                                    negative_answer_keys.append(key)

                            if not gold_answer_keys:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Error: Gold Answer Mapping", "Internal error: Could not map gold paragraph key to answer list.")
                                instances_skipped_missing_data += 1; continue


                            pi4pc_instance = {
                                "application_number": target_app_num,
                                "claim_number": claim_num,
                                "context": target_context,
                                "prior_art_specification": prior_art_details,
                                "options": options_dict,
                                "gold_answers": gold_answer_keys,
                                "silver_answers": silver_answer_keys,
                                "negative_answers": negative_answer_keys
                            }

                            output_filename = f"pi4pc_p_{pc_file.stem}_cl{claim_num}_{cited_patent_id}_{chosen_gold_key}.json"
                            output_filepath = output_dir / output_filename
                            try:
                                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                                    json.dump(pi4pc_instance, f_out, ensure_ascii=False, indent=2)
                                pi4pc_generated_count += 1
                                all_pi4pc_data.append(pi4pc_instance) 
                            except Exception as e_save:
                                log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Error: Save PI4PC", f"Failed to save PI4PC instance {output_filename}: {e_save}")


                        except Exception as e_inner:
                             log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, claim_num, cited_patent_id, "Error: Processing Instance", f"Unhandled exception during PI4PC instance creation: {e_inner}\n{traceback.format_exc(limit=1)}")
                             instances_skipped_missing_data += 1

            except Exception as e_outer_file:
                 log_status(log_writer, log_csv_file, datetime.now().isoformat(), pc_file.name, None, None, "Error: Processing File", f"Unhandled exception during processing file {pc_file.name}: {e_outer_file}\n{traceback.format_exc(limit=1)}")
                 files_skipped_load_error += 1

    finally:
        if log_csv_file: log_csv_file.close()
        print("\nLog file closed.")

    print("\n--- Cited Paragraph PI4PC Benchmark Generation Summary ---")
    print(f"Total pC files attempted: {total_pc_files}")
    print(f"pC files successfully processed (loaded record & pC): {files_processed}")
    if files_skipped_load_error > 0: print(f"pC files skipped due to loading/parsing errors or missing target context: {files_skipped_load_error}")
    print(f"PI4PC instances skipped due to missing essential data (prior art context, specs, keys, content): {instances_skipped_missing_data}")
    if instances_skipped_option_failure > 0: print(f"PI4PC instances skipped due to insufficient options generation: {instances_skipped_option_failure}")
    print(f"Total PI4PC instances generated and saved (individual files): {pi4pc_generated_count}")
    print(f"Logs saved to: {log_path}")

    if all_pi4pc_data and output_file_jsonl:
        print(f"\nSaving {len(all_pi4pc_data)} PI4PC instances to {output_file_jsonl}...")
        try:
            with open(output_file_jsonl, 'w', encoding='utf-8') as f_out:
                for entry in all_pi4pc_data:
                    json.dump(entry, f_out, ensure_ascii=False); f_out.write('\n')
            print(f"PI4PC data saved successfully to JSONL: {output_file_jsonl}")
            log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "JSONL Saved", f"Saved {len(all_pi4pc_data)} items to {output_file_jsonl}")
        except Exception as e_jsonl:
            print(f"Error writing PI4PC JSONL file {output_file_jsonl}: {e_jsonl}", file=sys.stderr)
            log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "Error: JSONL Save", str(e_jsonl))
    elif not all_pi4pc_data:
        print("\nNo PI4PC data generated to save to JSONL/Parquet.")
        if log_writer: log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "Info: No aggregated data", "PI4PC data list is empty.")

    if all_pi4pc_data and output_file_parquet:
        print(f"\nSaving {len(all_pi4pc_data)} PI4PC instances to {output_file_parquet}...")
        try:
            df_pi4pc = pd.DataFrame(all_pi4pc_data)
            for col in ['context', 'prior_art_specification', 'options']:
                if col in df_pi4pc.columns:
                    needs_serialization = df_pi4pc[col].apply(lambda x: isinstance(x, (dict, list))).any()
                    if needs_serialization:
                        try:
                            df_pi4pc[col] = df_pi4pc[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
                        except Exception as e_serial:
                            print(f"Warning: Could not serialize column '{col}' to JSON string for Parquet: {e_serial}. Attempting direct save.", file=sys.stderr)
                            if log_writer: log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", f"Warning: Parquet Serialize {col}", str(e_serial))
            
            df_pi4pc.to_parquet(output_file_parquet, index=False, engine='pyarrow')
            print(f"PI4PC data saved successfully to Parquet: {output_file_parquet}")
            log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "Parquet Saved", f"Saved {len(all_pi4pc_data)} items to {output_file_parquet}")
        except ImportError:
            print("Error: 'pandas' and/or 'pyarrow' are required to save to Parquet. Please install them.", file=sys.stderr)
            if log_writer: log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "Error: Parquet Lib Missing", "pandas or pyarrow not found")
        except Exception as e_parquet:
            print(f"Error writing PI4PC Parquet file {output_file_parquet}: {e_parquet}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if log_writer: log_status(log_writer, log_csv_file, datetime.now().isoformat(), "AGGREGATED", "ALL", "N/A", "Error: Parquet Save", str(e_parquet))

if __name__ == "__main__":
    try:
        import pandas as pd
        import pyarrow
    except ImportError:
        print("FATAL ERROR: 'pandas' and 'pyarrow' libraries are required for Parquet output.", file=sys.stderr)
        print("Please install them using: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    base_data_dir = script_dir.parent / "data"

    if not base_data_dir.is_dir():
        print(f"FATAL ERROR: Base data directory not found at expected location: {base_data_dir}", file=sys.stderr)
        sys.exit(1)

    paragraph_input_dir = base_data_dir / 'parsed_CTNF_with_PN'
    spec_input_dir = base_data_dir / 'spec_cited' / 'text' / 'parsed'
    record_input_dir = base_data_dir / 'record_with_title'
    output_individual_pi4pc_dir = base_data_dir / 'pi4pc'
    error_report_dir = base_data_dir / 'error_report'

    output_jsonl_file = base_data_dir / "pi4pc" / "pi4pc_benchmark.jsonl"
    output_parquet_file = base_data_dir / "pi4pc" / "pi4pc_benchmark.parquet"

    create_cited_paragraph_pi4pc(
        paragraph_dir=paragraph_input_dir,
        spec_dir=spec_input_dir,
        record_dir=record_input_dir,
        output_dir=output_individual_pi4pc_dir, 
        error_dir=error_report_dir,
        output_file_jsonl=output_jsonl_file,
        output_file_parquet=output_parquet_file
    )
    print("\n--- Script Finished ---")

