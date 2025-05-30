import json
import csv
from datetime import datetime
from pathlib import Path
import random
import re
import traceback
import sys
import pandas as pd
import pyarrow


def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"JSON Decode Error: {filepath}")
        return None
    except Exception as e:
        print(f"File Read Error: {filepath}: {e}")
        return None

def parse_date(date_string):
    if not date_string or not isinstance(date_string, str):
        return None
    try:
        date_string_cleaned = date_string.split('.')[0]
        return datetime.fromisoformat(date_string_cleaned.replace('Z', '+00:00'))
    except ValueError:
        pass
    try:
        return datetime.strptime(date_string.split('T')[0], '%Y-%m-%d')
    except ValueError:
        pass
    return None

def get_main_class(record_data):
    patent_class_full = record_data.get("class", "")
    if patent_class_full and isinstance(patent_class_full, str):
        return patent_class_full.split('/')[0]
    return None

def extract_cited_patent_ids(cited_list, id_key="publicationNumber"):
    ids = set()
    if not isinstance(cited_list, list):
        return list(ids)
    for item in cited_list:
        patent_id = None
        if isinstance(item, dict):
            patent_id = item.get(id_key) or item.get("patentDocumentIdentifier")
        elif isinstance(item, str):
             patent_id = item
        if patent_id:
             normalized_id = re.sub(r'[^a-zA-Z0-9]', '', str(patent_id)).upper()
             if normalized_id.isdigit() and len(normalized_id) >= 7:
                 normalized_id = "US" + normalized_id
             elif not re.match(r'^[A-Z]{2}', normalized_id) and len(normalized_id) >= 7:
                 normalized_id = "US" + normalized_id
             ids.add(normalized_id)
    return list(ids)

def get_patent_details_from_record_examiner_citations(patent_id_to_find, record_data):
    default_details = {
        "patent_id": patent_id_to_find,
        "title": "N/A",
        "abstract": "N/A",
        "claims": []
    }
    if not record_data or "patentsCitedByExaminer" not in record_data or not isinstance(record_data["patentsCitedByExaminer"], list):
        return default_details

    for cited_patent in record_data["patentsCitedByExaminer"]:
        if isinstance(cited_patent, dict):
            ref_id = cited_patent.get("referenceIdentifier")
            normalized_ref_id = None
            if ref_id:
                 normalized_ref_id = re.sub(r'[^a-zA-Z0-9]', '', str(ref_id)).upper()
                 if normalized_ref_id.isdigit() and len(normalized_ref_id) >= 7:
                     normalized_ref_id = "US" + normalized_ref_id
                 elif not re.match(r'^[A-Z]{2}', normalized_ref_id) and len(normalized_ref_id) >= 7:
                      normalized_ref_id = "US" + normalized_ref_id

            if normalized_ref_id and normalized_ref_id == patent_id_to_find:
                return {
                    "patent_id": patent_id_to_find,
                    "title": cited_patent.get("title", "N/A"),
                    "abstract": cited_patent.get("abstract", "N/A"),
                    "claims": cited_patent.get("claims", [])
                }
    return default_details

def find_rejected_102_103_claims(ctnf_data):
    rejected_claim_numbers = []
    if not isinstance(ctnf_data, dict) or "claims" not in ctnf_data: return rejected_claim_numbers
    for claim in ctnf_data["claims"]:
        if not isinstance(claim, dict): continue
        claim_num = claim.get("claimNumber"); is_rejected = claim.get("isReject", False)
        if is_rejected:
            valid_rejection_found = False
            has_cited_patents = False
            reasons = claim.get("reasons", []);
            if not isinstance(reasons, list): continue
            for reason in reasons:
                if not isinstance(reason, dict): continue
                section_code = reason.get("sectionCode", ""); section_code_str = str(section_code)
                is_102_or_103 = ("102" in section_code_str or "103" in section_code_str)
                if is_102_or_103:
                    valid_rejection_found = True
                    cited_patents = reason.get("citedPatents", [])
                    if cited_patents and isinstance(cited_patents, list):
                        if extract_cited_patent_ids(cited_patents):
                            has_cited_patents = True
            if valid_rejection_found and has_cited_patents:
                 rejected_claim_numbers.append(claim_num)
    return rejected_claim_numbers

def get_gold_silver_ids(ctnf_data, target_claim_number):
    gold_ids = set(); silver_pool = set()
    if not isinstance(ctnf_data, dict) or "claims" not in ctnf_data: return list(gold_ids), list(silver_pool)
    for claim in ctnf_data["claims"]:
        if not isinstance(claim, dict): continue
        claim_num = claim.get("claimNumber"); reasons = claim.get("reasons", [])
        if not isinstance(reasons, list): continue
        claim_citations = set()
        for reason in reasons:
             if not isinstance(reason, dict): continue
             claim_citations.update(extract_cited_patent_ids(reason.get("citedPatents", [])))
        if str(claim_num) == str(target_claim_number): gold_ids.update(claim_citations)
        else: silver_pool.update(claim_citations)
    silver_pool = silver_pool - gold_ids
    return list(gold_ids), list(silver_pool)

def find_negative_candidates_pool(current_index, target_class, current_filing_date,
                                  all_record_files, record_dir, pc_dir, exclude_ids, target_pool_size=50):
    negative_pool = set()
    processed_app_ids = set()
    neg_id_to_source_record_data = {}

    current_record_info = all_record_files[current_index]
    current_app_num = current_record_info["app_num"]
    current_filing_dt = parse_date(current_filing_date)

    print(f"  [Neg Pool Search] Starting: App={current_app_num}, Target Class={target_class}, Target Date < {current_filing_dt.date() if current_filing_dt else 'N/A'}, Target Pool Size={target_pool_size}")
    if not current_filing_dt:
         print("    [Neg Pool Search] Warning: Current filingDate is invalid. Cannot proceed with Filing Date comparison. Pool creation is not possible.")
         return [], {}

    forward_range = range(current_index + 1, len(all_record_files))
    backward_range = range(0, current_index)
    search_indices = list(forward_range) + list(backward_range)

    print(f"    [Neg Pool Search] Target Record Indices ({len(search_indices)}): {search_indices[:5]}...{search_indices[-5:] if len(search_indices) > 10 else ''}")

    for i in search_indices:
        if len(negative_pool) >= target_pool_size:
            print(f"    [Neg Pool Search] Reached target pool size ({target_pool_size}). Search halted.")
            break

        next_record_info = all_record_files[i]
        next_rec_num = next_record_info["rec_num"]; next_app_num = next_record_info["app_num"]

        if next_app_num == current_app_num or next_app_num in processed_app_ids: continue
        processed_app_ids.add(next_app_num)

        print(f"    [Neg Pool Search] Checking (index {i}): Record r{next_rec_num:05d}_{next_app_num}")

        next_record_file = next(record_dir.glob(f"rec_r{next_rec_num:05d}_{next_app_num}.json"), None)
        if not next_record_file: print(f"      [Neg Pool Search] Skipped: Record file not found"); continue
        next_record_data = load_json(next_record_file)
        if not next_record_data: print(f"      [Neg Pool Search] Skipped: Failed to load record file"); continue

        next_class = get_main_class(next_record_data)
        if not next_class: print(f"      [Neg Pool Search] Skipped: No main class in next record"); continue
        if next_class == target_class: print(f"      [Neg Pool Search] Skipped: Same class ({next_class})"); continue

        next_filing_date_str = next_record_data.get("filingDate")
        next_filing_dt = parse_date(next_filing_date_str)

        is_suitable_source = False
        log_reason = ""
        if not next_filing_dt: log_reason = f"Failed to parse next record Filing Date ({next_filing_date_str})"
        elif not (next_filing_dt < current_filing_dt): log_reason = f"Filing Date ({next_filing_dt.date()}) >= Current ({current_filing_dt.date()})"
        else: is_suitable_source = True

        if is_suitable_source:
            print(f"      [Neg Pool Search] ✅ Suitable source found! (Class: {next_class}, FilingDate: {next_filing_dt.date()})")
            potential_negatives_from_source = set()
            cited_by_examiner = next_record_data.get("patentsCitedByExaminer", [])
            examiner_ids = extract_cited_patent_ids(cited_by_examiner, "referenceIdentifier")
            if examiner_ids: print(f"        [Neg Pool Search] Source contains Examiner citations ({len(examiner_ids)}): {examiner_ids[:5]}..."); potential_negatives_from_source.update(examiner_ids)
            else: print(f"        [Neg Pool Search] Source contains no Examiner citations")
            pc_matches = list(Path(pc_dir).glob(f"pC_*_{next_app_num}.json"))
            pc_ids = set()
            if pc_matches:
                pc_data_neg_src = load_json(pc_matches[0])
                if pc_data_neg_src and "claims" in pc_data_neg_src:
                    for claim in pc_data_neg_src["claims"]:
                         if isinstance(claim, dict) and "reasons" in claim:
                             for reason in claim.get("reasons", []):
                                 if isinstance(reason, dict) and "citedPatents" in reason:
                                     pc_ids.update(extract_cited_patent_ids(reason.get("citedPatents", [])))
            if pc_ids: print(f"        [Neg Pool Search] Source contains pC citations ({len(pc_ids)}): {list(pc_ids)[:5]}..."); potential_negatives_from_source.update(pc_ids)
            else: print(f"        [Neg Pool Search] Source contains no pC citations (or pC file/data issue)")

            for neg_id in potential_negatives_from_source:
                if neg_id not in neg_id_to_source_record_data:
                    neg_id_to_source_record_data[neg_id] = next_record_data

            new_candidates = potential_negatives_from_source - exclude_ids - negative_pool
            if new_candidates:
                 print(f"        [Neg Pool Search] ➡️ {len(new_candidates)} new candidates added (total source citations: {len(potential_negatives_from_source)}, excluding excluded IDs).")
                 negative_pool.update(new_candidates)
                 print(f"        [Neg Pool Search] Current pool size: {len(negative_pool)} / {target_pool_size}")
            else:
                 print(f"        [Neg Pool Search] ⚠️ No new candidates (all source citations already excluded or in pool).")
        else:
            print(f"      [Neg Pool Search] Skipped: {log_reason}")

    print(f"  [Neg Pool Search] Final: Total {len(negative_pool)} negative candidates pool obtained.")
    return list(negative_pool), neg_id_to_source_record_data


def create_par4pc_data(record_dir: Path, ctnf_dir: Path, output_dir: Path, error_dir: Path, output_file_jsonl: Path, output_file_parquet: Path):
    try:
        print(f"Record directory path: {record_dir}")
        if not record_dir.is_dir(): print(f"Error: Record directory not found: {record_dir}"); return
        if not ctnf_dir.is_dir(): print(f"Error: CTNF directory not found: {ctnf_dir}"); return

    except NameError:
        print("Error: Unable to check paths (NameError). This should ideally not happen with direct path parameters.");
        return

    output_dir.mkdir(parents=True, exist_ok=True); error_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file_jsonl: output_file_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_file_parquet: output_file_parquet.parent.mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_log_path = error_dir / f"error_par4pc_generation_{current_time}.csv"; log_fieldnames = ["rec_num", "app_num", "claim_num", "error_code", "error_message", "details"]
    with open(error_log_path, 'w', newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=log_fieldnames).writeheader()

    def log_status(rec_num, app_num, claim_num, code=-1, msg="", details=""):
         details_str = str(details); details_str = details_str[:1000]+"..." if len(details_str) > 1000 else details_str
         try:
             with open(error_log_path, 'a', newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=log_fieldnames).writerow({"rec_num": rec_num, "app_num": app_num, "claim_num": claim_num if claim_num else "N/A", "error_code": code, "error_message": msg, "details": details_str})
         except Exception as log_e: print(f"Error: Log recording failed: {log_e}")

    # 모든 레코드 파일을 찾아서 유효한 것으로 처리
    record_files = sorted(list(record_dir.glob("rec_*.json")))
    if not record_files: 
        print("Error: No record files found."); 
        log_status("N/A","N/A",None,404,"No record files found", str(record_dir)); 
        return
    
    all_records = []
    for record_file in record_files:
        match = re.search(r'rec_(r\d+)_(\d+)', record_file.name)
        if match:
            rec_num = int(match.group(1)[1:])  # 'r123' -> 123
            app_num = match.group(2)
            all_records.append({"rec_num": rec_num, "app_num": app_num})
    
    if not all_records: 
        print("Error: No valid record file names found."); 
        return
    
    print(f"Number of record files: {len(all_records)}"); 
    all_records.sort(key=lambda x: x["rec_num"])
    
    all_par4pc_data = []
    par4pc_count = 0; num_options = 8; negative_pool_target_size = 100 
    min_negatives = 3; max_gold_silver = num_options - min_negatives

    for idx, record_info in enumerate(all_records):
        rec_num = record_info["rec_num"]; app_num = record_info["app_num"]
        print(f"\n--- Processing started: Record r{rec_num:05d} (App: {app_num}) ---")
        try:
            record_file = next(record_dir.glob(f"rec_r{rec_num:05d}_{app_num}.json"), None)
            ctnf_files = list(ctnf_dir.glob(f"pC_*_{app_num}.json"))
            if not record_file: log_status(rec_num, app_num, None, 404, "Record file not found"); print("  Error: Record file not found"); continue
            
            
            if not ctnf_files:
                record_file_match = re.search(r'rec_(r\d+)_', record_file.name)
                if record_file_match:
                    r_index = record_file_match.group(1)
                    ctnf_files = list(ctnf_dir.glob(f"pC_{r_index}_{app_num}.json"))
            
            if not ctnf_files: log_status(rec_num, app_num, None, 404, "CTNF file not found"); print("  Error: CTNF file not found"); continue

            ctnf_file = ctnf_files[0];
            if len(ctnf_files) > 1: print(f"  Warning: Multiple CTNF files found. Using first: {ctnf_files[0]}")
            record_data = load_json(record_file); ctnf_data = load_json(ctnf_file)
            if not record_data: log_status(rec_num, app_num, None, 500, "Failed to load record data"); print("  Error: Failed to load record data"); continue
            if not ctnf_data: log_status(rec_num, app_num, None, 500, "Failed to load CTNF data"); print("  Error: Failed to load CTNF data"); continue

            
            examiner_citations = record_data.get("patentsCitedByExaminer", [])
            unique_examiner_cited_ids = extract_cited_patent_ids(examiner_citations, "referenceIdentifier")
            examiner_cited_count = len(unique_examiner_cited_ids)
            print(f"  Examiner citation count (unique): {examiner_cited_count}")
            if not (1 <= examiner_cited_count <= 5): print(f"  Skipped: Examiner citation count ({examiner_cited_count}) not in [1, 5]"); log_status(rec_num, app_num, None, 312, f"Skipped: Examiner citation count ({examiner_cited_count}) not in [1, 5]"); continue

            
            contains_other_rejection = False; has_any_rejected_claim = False
            if "claims" in ctnf_data and isinstance(ctnf_data["claims"], list):
                for claim in ctnf_data["claims"]:
                    if not isinstance(claim, dict): continue
                    if claim.get("isReject", False):
                        has_any_rejected_claim = True; reasons = claim.get("reasons", [])
                        if not isinstance(reasons, list) or not reasons: log_status(rec_num, app_num, claim.get('claimNumber'), 314, "Rejected claim has no reasons listed"); continue
                        for reason in reasons:
                             if not isinstance(reason, dict): continue
                             section_code = reason.get("sectionCode", ""); section_code_str = str(section_code).strip()
                             codes_in_reason = set(filter(None, re.split(r'\D+', section_code_str)))
                             if not codes_in_reason: continue
                             if any(code not in {'102', '103'} for code in codes_in_reason): contains_other_rejection = True; print(f"  건너뜀 조건 발견: 청구항 {claim.get('claimNumber')} 이유 코드 '{section_code_str}'가 102/103 외 포함."); break
                        if contains_other_rejection: break
            if contains_other_rejection or not has_any_rejected_claim:
                 log_msg = "Skipped: Contains non-102/103 rejection reasons" if contains_other_rejection else "Skipped: No rejected claims found"
                 log_code = 313 if contains_other_rejection else 315; print(f"  Skipped: {log_msg}"); log_status(rec_num, app_num, None, log_code, log_msg); continue

            print("  ✅ Passed filters: Examiner citations 1-5 & only 102/103 rejections")
            target_claim_numbers = find_rejected_102_103_claims(ctnf_data)
            if not target_claim_numbers: print("  Error/Warning: Passed filters but no claims identified?"); log_status(rec_num, app_num, None, 300, "Filters passed but no claims identified?"); continue
            print(f"  Target claim numbers (102/103 rejections): {target_claim_numbers}")

            target_class = get_main_class(record_data); current_filing_date = record_data.get("filingDate")

            
            all_gold_ids_app = set()
            all_silver_ids_app_pool = set()
            for claim_num_for_ids in target_claim_numbers:
                gold_ids_for_claim, silver_ids_for_claim = get_gold_silver_ids(ctnf_data, claim_num_for_ids)
                all_gold_ids_app.update(gold_ids_for_claim)
                all_silver_ids_app_pool.update(silver_ids_for_claim)
            all_silver_ids_app_pool = all_silver_ids_app_pool - all_gold_ids_app
            print(f"  App-wide Gold ID candidates ({len(all_gold_ids_app)}): {list(all_gold_ids_app)[:10]}...")
            print(f"  App-wide Silver ID candidates ({len(all_silver_ids_app_pool)}): {list(all_silver_ids_app_pool)[:10]}...")


            negative_candidates_pool_ids = []
            neg_id_sources_map = {}
            if target_class and current_filing_date:
                
                all_app_citations_to_exclude = set(unique_examiner_cited_ids) | all_gold_ids_app | all_silver_ids_app_pool
                negative_candidates_pool_ids, neg_id_sources_map = find_negative_candidates_pool(
                    idx, target_class, current_filing_date,
                    all_records, record_dir, ctnf_dir, all_app_citations_to_exclude, 
                    target_pool_size=negative_pool_target_size
                )
            elif not target_class: log_status(rec_num, app_num, None, 306, "Target class not found"); print("  Warning: No target class found")
            elif not current_filing_date: log_status(rec_num, app_num, None, 307, "Filing date not found"); print("  Warning: Filing date not found")

            if not negative_candidates_pool_ids: print("  경고: 네거티브 후보 풀 없음."); log_status(rec_num, app_num, None, 308, "Negative candidate pool is empty for the application")


            fixed_final_gold_ids = []; fixed_final_silver_ids = []; fixed_final_negative_ids = []
            fixed_all_option_ids = []

            eligible_app_gold_ids = list(all_gold_ids_app); random.shuffle(eligible_app_gold_ids)
            if len(eligible_app_gold_ids) > max_gold_silver: fixed_final_gold_ids = eligible_app_gold_ids[:max_gold_silver]
            else: fixed_final_gold_ids = eligible_app_gold_ids
            print(f"  Fixed option ID selection - Gold ID sampling: {len(fixed_final_gold_ids)} (app-wide candidates: {len(all_gold_ids_app)})")
      
            needed_for_gs = max_gold_silver - len(fixed_final_gold_ids)
            if needed_for_gs > 0 and all_silver_ids_app_pool:
                eligible_app_silver_ids = list(all_silver_ids_app_pool - set(fixed_final_gold_ids)); random.shuffle(eligible_app_silver_ids)
                take_silver = min(needed_for_gs, len(eligible_app_silver_ids)); fixed_final_silver_ids = eligible_app_silver_ids[:take_silver]
                print(f"  Fixed option ID selection - Silver ID sampling: {len(fixed_final_silver_ids)} (app-wide candidates: {len(all_silver_ids_app_pool)})")

            current_gs_count = len(fixed_final_gold_ids) + len(fixed_final_silver_ids)
            negatives_needed = num_options - current_gs_count
            if negatives_needed < min_negatives:
                print(f"  Error: App '{app_num}' failed to create fixed options - needed negatives ({negatives_needed}) < minimum ({min_negatives}). Skipping record."); log_status(rec_num, app_num, None, 503, f"Cannot create fixed options: Negatives needed ({negatives_needed}) < min ({min_negatives})"); continue

            if negatives_needed > 0 and negative_candidates_pool_ids:
                eligible_negative_ids = list(set(negative_candidates_pool_ids) - set(fixed_final_gold_ids) - set(fixed_final_silver_ids))
                if len(eligible_negative_ids) < negatives_needed:
                    print(f"  Error: App '{app_num}' failed to create fixed options - insufficient negatives ({len(eligible_negative_ids)}) < needed ({negatives_needed}). Skipping record."); log_status(rec_num, app_num, None, 309, f"Cannot create fixed options: Insufficient neg candidates from pool: found {len(eligible_negative_ids)}, needed {negatives_needed}"); continue
                else:
                    random.shuffle(eligible_negative_ids); fixed_final_negative_ids = eligible_negative_ids[:negatives_needed]
                    print(f"  Fixed option ID selection - Negative ID sampling: {len(fixed_final_negative_ids)} (pool: {len(negative_candidates_pool_ids)}, available: {len(eligible_negative_ids)})")
            elif negatives_needed > 0:
                print(f"  Error: App '{app_num}' failed to create fixed options - needed negatives ({negatives_needed}) but pool is empty. Skipping record."); log_status(rec_num, app_num, None, 310, f"Cannot create fixed options: Neg needed ({negatives_needed}) but pool empty"); continue

            fixed_all_option_ids = fixed_final_gold_ids + fixed_final_silver_ids + fixed_final_negative_ids
            if len(fixed_all_option_ids) != num_options:
                print(f"  Error: App '{app_num}' failed to create fixed options - final option ID count ({len(fixed_all_option_ids)}) != {num_options}. Skipping record."); log_status(rec_num, app_num, None, 504, f"Cannot create fixed options: Final option count {len(fixed_all_option_ids)} != {num_options}"); continue
            random.shuffle(fixed_all_option_ids)
            print(f"  Fixed option ID list created (8 options): {fixed_all_option_ids}")

            fixed_options_dict = {}
            fixed_id_to_letter = {}
            print(f"  Looking up fixed option details (using rec_r{rec_num:05d}_{app_num}.json and negative sources)...")
            for i, option_id in enumerate(fixed_all_option_ids):
                letter = chr(ord('A') + i)
                option_details = get_patent_details_from_record_examiner_citations(option_id, record_data)
                if option_details["title"] == "N/A" and option_id in neg_id_sources_map:
                    print(f"    Option {letter} ({option_id}): Not found in current record. Trying negative source...")
                    source_record_data = neg_id_sources_map[option_id]
                    option_details = get_patent_details_from_record_examiner_citations(option_id, source_record_data)
                    if option_details["title"] != "N/A":
                         print(f"      -> Found details in negative source")
                    else:
                         print(f"      -> No details found in negative source")

                fixed_options_dict[letter] = option_details
                fixed_id_to_letter[option_details["patent_id"]] = letter

            print(f"  Fixed option details dictionary created (with details)")

            for target_claim in target_claim_numbers:
                print(f"\n  Generating par4pc for claim {target_claim}...")
                claim_gold_ids, claim_silver_ids_pool = get_gold_silver_ids(ctnf_data, target_claim)
                if not claim_gold_ids: print(f"    Warning: Claim {target_claim} has no Gold IDs (skipping par4pc generation)."); log_status(rec_num, app_num, target_claim, 302, "No Gold IDs found for this specific claim"); continue

                gold_letters = sorted([fixed_id_to_letter[pid] for pid in claim_gold_ids if pid in fixed_id_to_letter])
                claim_silver_ids = set(claim_silver_ids_pool) - set(claim_gold_ids)
                silver_letters = sorted([fixed_id_to_letter[pid] for pid in claim_silver_ids if pid in fixed_id_to_letter])
                
                current_option_ids_in_use = set(claim_gold_ids) | set(claim_silver_ids)
                negative_letters = sorted([
                    letter for letter, details_dict in fixed_options_dict.items() 
                    if details_dict.get("patent_id") not in current_option_ids_in_use
                ])


                if len(gold_letters) + len(silver_letters) + len(negative_letters) != num_options:
                     print(f"    Warning: Claim {target_claim} - After mapping, total letter count ({len(gold_letters) + len(silver_letters) + len(negative_letters)}) != {num_options}. Gold: {len(gold_letters)}, Silver: {len(silver_letters)}, Negative: {len(negative_letters)}");
                     log_status(rec_num, app_num, target_claim, 505, f"Answer letter count mismatch: G={len(gold_letters)}, S={len(silver_letters)}, N={len(negative_letters)}")
    
                par4pc_data_item = {
                    "application_number": str(app_num), "claim_number": str(target_claim),
                    "context": {"title": record_data.get("title", "N/A"), "abstract": record_data.get("abstract", "N/A"), "claims": record_data.get("initialClaims", [])},
                    "options": fixed_options_dict,
                    "gold_answers": gold_letters,
                    "silver_answers": silver_letters,
                    "negative_answers": negative_letters
                }
                
                
                individual_output_file = output_dir / f"par4pc_r{rec_num:05d}_{app_num}_cl{target_claim}.json"
                try:
                    with open(individual_output_file, 'w', encoding='utf-8') as f: json.dump(par4pc_data_item, f, indent=2, ensure_ascii=False)
                    print(f"    ✅ par4pc (individual) saved: {individual_output_file}"); log_status(rec_num, app_num, target_claim, -1, "Success (individual JSON)")
                    all_par4pc_data.append(par4pc_data_item)
                    par4pc_count += 1
                except IOError as e: print(f"    Error: Failed to save par4pc file: {e}"); log_status(rec_num, app_num, target_claim, 502, f"Failed to save par4pc file: {e}")
        except Exception as e: print(f"    Error: Exception occurred while processing application {app_num}: {e}"); log_status(rec_num, app_num, None, 400, f"Unhandled exception: {e}", traceback.format_exc(limit=1))

    print(f"\n--- Final completion ---"); print(f"Total individual par4pcs generated: {par4pc_count}")

    
    if all_par4pc_data and output_file_jsonl:
        print(f"\nSaving {len(all_par4pc_data)} par4pc instances to {output_file_jsonl}...")
        try:
            with open(output_file_jsonl, 'w', encoding='utf-8') as f_out:
                for entry in all_par4pc_data:
                    json.dump(entry, f_out, ensure_ascii=False); f_out.write('\n')
            print(f"par4pc data saved successfully to JSONL: {output_file_jsonl}")
            log_status("N/A", "N/A", "ALL", -1, "JSONL Saved", f"Saved {len(all_par4pc_data)} items to {output_file_jsonl}")
        except Exception as e_jsonl:
             print(f"Error writing par4pc JSONL file {output_file_jsonl}: {e_jsonl}", file=sys.stderr)
             log_status("N/A", "N/A", "ALL", 601, "JSONL Save Error", str(e_jsonl))
    elif not all_par4pc_data:
        print("\nNo par4pc data generated to save to JSONL/Parquet.")
        log_status("N/A", "N/A", "ALL", 0, "No data for aggregated files", "par4pc data list is empty.")

    
    if all_par4pc_data and output_file_parquet:
        print(f"\nSaving {len(all_par4pc_data)} par4pc instances to {output_file_parquet}...")
        try:
            df_par4pc = pd.DataFrame(all_par4pc_data)

            for col in ['context', 'options']:
                if col in df_par4pc.columns:
                     needs_serialization = df_par4pc[col].apply(lambda x: isinstance(x, (dict, list))).any()
                     if needs_serialization:
                         try:
                             df_par4pc[col] = df_par4pc[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
                         except Exception as e_serial:
                             print(f"Warning: Could not serialize column '{col}' to JSON string for Parquet: {e_serial}. Attempting direct save.", file=sys.stderr)
                             log_status("N/A", "N/A", "ALL", 603, f"Parquet column serialization warning: {col}", str(e_serial))
            
            df_par4pc.to_parquet(output_file_parquet, index=False, engine='pyarrow')
            print(f"par4pc data saved successfully to Parquet: {output_file_parquet}")
            log_status("N/A", "N/A", "ALL", -1, "Parquet Saved", f"Saved {len(all_par4pc_data)} items to {output_file_parquet}")
        except ImportError:
            print("Error: 'pandas' and/or 'pyarrow' are required to save to Parquet. Please install them.", file=sys.stderr)
            log_status("N/A", "N/A", "ALL", 600, "Parquet Save Error: Missing Library", "pandas or pyarrow not found")
        except Exception as e_parquet:
            print(f"Error writing par4pc Parquet file {output_file_parquet}: {e_parquet}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            log_status("N/A", "N/A", "ALL", 602, "Parquet Save Error", str(e_parquet))

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_data_dir = script_dir.parent / "data"

    if not base_data_dir.is_dir():
        print(f"FATAL ERROR: Base data directory not found at expected location: {base_data_dir}", file=sys.stderr)
        sys.exit(1)
  
    record_input_dir = base_data_dir / "record"
    ctnf_input_dir = base_data_dir / "parsed_CTNF_with_PN"
    individual_json_output_dir = base_data_dir / "par4pc" 
    error_report_dir = base_data_dir / "error_report"
    output_jsonl_file = base_data_dir / "par4pc" / "par4pc_benchmark.jsonl"
    output_parquet_file = base_data_dir / "par4pc" / "par4pc_benchmark.parquet"
  
    create_par4pc_data(
        record_dir=record_input_dir,
        ctnf_dir=ctnf_input_dir,
        output_dir=individual_json_output_dir,
        error_dir=error_report_dir,
        output_file_jsonl=output_jsonl_file,
        output_file_parquet=output_parquet_file
    )
    print("\n--- Script Finished ---")