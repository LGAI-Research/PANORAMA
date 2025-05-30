# python PANORAMA/convert2bench_noc4pc.py

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import traceback
from datetime import datetime
import csv
import sys
import pandas as pd
from collections import defaultdict


def extract_core_patent_number(raw_num: str) -> str | None:
    """Extracts the core numeric/alphanumeric part from patent numbers/identifiers."""
    if not isinstance(raw_num, str):
        return None
    cleaned = re.sub(r'^(US|EP|WO|JP|KR)\s*|\s|,|-', '', raw_num.upper())
    match = re.search(r'([\dX]+)$', cleaned)
    if match:
        return match.group(1)
    if re.fullmatch(r'[\dX]+', cleaned):
        return cleaned
    digits = re.findall(r'[\dX]', cleaned)
    if digits:
        return "".join(digits)
    return None

def find_spec_file(spec_dir: Path, cited_patent_num_cleaned: str) -> Path | None:
    """Finds the specification file based on the cleaned patent number."""
    if not cited_patent_num_cleaned: return None
    target_filename_pattern = f"spec_txt_{cited_patent_num_cleaned}_parsed.json"
    potential_file = spec_dir / target_filename_pattern


    if potential_file.is_file():
        return potential_file
    return None

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

def log_status(writer, file_handle, timestamp, record_file, pc_file, claim_num, status, details):
    entry = {
        'Timestamp': timestamp,
        'Record_File': record_file if record_file else 'N/A',
        'pC_File': pc_file if pc_file else 'N/A',
        'Claim_Number': claim_num if claim_num else 'ALL',
        'Status': status,
        'Details': str(details)[:2000]
    }
    try:
        if writer and file_handle:
             writer.writerow(entry)
             file_handle.flush()
             return 1
        else:
             print(f"ERROR: CSV Writer or File Handle not initialized. Log entry lost: {entry}", file=sys.stderr)
             return 0
    except Exception as e:
        print(f"Error writing log entry: {e} - Entry: {entry}", file=sys.stderr)
        try:
             if file_handle: file_handle.flush()
        except Exception: pass
        return 0

def parse_paragraph_key(raw_key: Any) -> int | None:
    """Attempts to parse various paragraph key formats into an integer."""
    if isinstance(raw_key, int):
        return raw_key
    if not isinstance(raw_key, str):
        return None
    try:
        return int(raw_key)
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
    return None # Return None if no parsing strategy worked

def create_rejection_benchmark(record_dir: Path, pc_dir: Path, spec_dir: Path, output_file_jsonl: Path, output_file_parquet: Path, individual_json_dir: Path, error_dir: Path):
    print(f"--- Rejection Benchmark Generation Started ---")
    print(f"Record Source: {record_dir}")
    print(f"pC Source (Rejection Info): {pc_dir}")
    print(f"Specification Source: {spec_dir}")
    print(f"JSONL Output: {output_file_jsonl}")
    print(f"Parquet Output: {output_file_parquet}")
    print(f"Individual JSON Output Dir: {individual_json_dir}")
    print(f"Error/Log Report Dir: {error_dir}")

    try:
        record_dir.mkdir(parents=True, exist_ok=True)
        pc_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)
        output_file_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_file_parquet.parent.mkdir(parents=True, exist_ok=True)
        individual_json_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured necessary directories exist.")
    except OSError as e:
        print(f"FATAL ERROR: Could not create necessary directories: {e}. Aborting.", file=sys.stderr)
        return

    if not record_dir.is_dir(): print(f"FATAL ERROR: Record data directory not found: {record_dir}", file=sys.stderr); return
    if not pc_dir.is_dir(): print(f"FATAL ERROR: pC (rejection) data directory not found: {pc_dir}", file=sys.stderr); return
    if not spec_dir.is_dir(): print(f"FATAL ERROR: Specification data directory not found: {spec_dir}", file=sys.stderr); return

    record_files = sorted(list(record_dir.glob("rec_*.json")))
    if not record_files: print(f"❌ ERROR: No rec_*.json files found in {record_dir}", file=sys.stderr); return
    total_records = len(record_files)
    print(f"Found {total_records} record files to process.")

    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_path = error_dir / f'rejection_benchmark_log_{timestamp_log}.csv'
    log_csv_file_handle = None
    log_csv_writer = None
    log_fieldnames = ['Timestamp', 'Record_File', 'pC_File', 'Claim_Number', 'Status', 'Details']
    try:
        log_csv_file_handle = open(detailed_log_path, 'w', newline='', encoding='utf-8-sig')
        log_csv_writer = csv.DictWriter(log_csv_file_handle, fieldnames=log_fieldnames)
        log_csv_writer.writeheader()
        print(f"Detailed log file initialized: {detailed_log_path}")
    except IOError as e:
        print(f"FATAL ERROR: Could not initialize detailed log file {detailed_log_path}: {e}. Aborting.", file=sys.stderr)
        if log_csv_file_handle: log_csv_file_handle.close(); return

    overall_benchmarks_created = 0
    overall_records_processed = 0
    overall_records_failed_loading = 0
    overall_pc_files_not_found = 0
    overall_pc_files_failed_loading = 0
    overall_claims_processed = 0
    overall_claims_skipped_missing_data = 0
    overall_spec_lookup_failures = 0
    overall_cited_detail_lookup_failures = 0
    overall_log_entries_written = 0
    overall_individual_files_saved = 0
    overall_individual_files_failed = 0
    overall_benchmark_data: List[Dict[str, Any]] = [] # For aggregated output

    try:
        for i, record_file in enumerate(record_files):
            target_patent_number = None
            pc_file_path = None
            pc_data = None
            target_title = "N/A"; target_abstract = "N/A"; initial_claims = []
            record_load_error = False; pc_load_error = False
            processed_claims_this_record = 0

            print(f"\n--- [{i+1}/{total_records}] Processing Record: {record_file.name}")

            try:
                try:
                    with open(record_file, 'r', encoding='utf-8') as f_rec: record_data = json.load(f_rec)
                    target_patent_number = record_data.get("applicationNumber")
                    if not target_patent_number:
                        raise ValueError("'applicationNumber' missing in record file")
                    target_title = record_data.get("title", "N/A")
                    target_abstract = record_data.get("abstract", "N/A")
                    initial_claims = record_data.get("initialClaims", [])
                    if not isinstance(initial_claims, list):
                        overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, None, None, 'Warning: Invalid Initial Claims', f"'initialClaims' is not a list, defaulting to empty.")
                        initial_claims = []
                    if target_title == "N/A" or target_abstract == "N/A":
                         overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, None, None, 'Warning: Missing Context', f"Title: {target_title}, Abstract: {target_abstract}")

                except Exception as e_rec:
                    record_load_error = True; overall_records_failed_loading += 1
                    error_detail = f"Loading record file: {type(e_rec).__name__}: {e_rec}"
                    overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, None, None, 'Error: Loading Record', error_detail)
                    print(f"  ❌ ERROR: {error_detail}. Skipping record.", file=sys.stderr); continue # Skip this record file

                match_r_index = re.search(r'rec_(r\d+)_', record_file.name)
                r_index = match_r_index.group(1) if match_r_index else "rUNKNOWN"
                pc_file_pattern = f"pC_{r_index}_{target_patent_number}.json"
                possible_pc_files = list(pc_dir.glob(pc_file_pattern))

                if not possible_pc_files:
                    overall_pc_files_not_found += 1
                    error_detail = f"pC file matching '{pc_file_pattern}' not found"
                    overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_pattern, None, 'Error: pC File Not Found', error_detail)
                    print(f"  ❌ ERROR: {error_detail}. Skipping record.", file=sys.stderr); continue # Skip this record file
                if len(possible_pc_files) > 1:
                     warn_detail = f"Multiple ({len(possible_pc_files)}) pC files found for '{pc_file_pattern}'. Using first: {possible_pc_files[0].name}"
                     overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_pattern, None, 'Warning: Multiple pC Files', warn_detail)
                pc_file_path = possible_pc_files[0]

                try:
                    with open(pc_file_path, 'r', encoding='utf-8') as f_pc: pc_data = json.load(f_pc)
                    claims_in_pc = pc_data.get("claims", [])
                    if not isinstance(claims_in_pc, list): raise ValueError("'claims' field is not a list in pC file")
                except Exception as e_pc:
                    pc_load_error = True; overall_pc_files_failed_loading += 1
                    error_detail = f"Loading pC file {pc_file_path.name}: {type(e_pc).__name__}: {e_pc}"
                    overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Error: Loading pC File', error_detail)
                    print(f"  ❌ ERROR: {error_detail}. Skipping record.", file=sys.stderr); continue # Skip this record file

                overall_records_processed += 1

                cited_patent_aggregated_data = defaultdict(lambda: {"raw_nums": set(), "cited_texts": set()})
                for claim_info_pc in claims_in_pc:
                    if not isinstance(claim_info_pc, dict): continue
                    reasons = claim_info_pc.get("reasons", [])
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
                                    cited_patent_aggregated_data[cleaned_cited_num]["raw_nums"].add(str(raw_cited_num))
                                    for key_raw in text_keys_raw:
                                        parsed_key = parse_paragraph_key(key_raw)
                                        if parsed_key is not None:
                                            cited_patent_aggregated_data[cleaned_cited_num]["cited_texts"].add(parsed_key)

                prior_art_specifications_list = []
                for cleaned_cited_num, agg_data in cited_patent_aggregated_data.items():
                    spec_lookup_failed_this_patent = False
                    citation_skipped = False

                    cited_details = find_cited_details_in_record(record_data, cleaned_cited_num)
                    if cited_details["title"] == "N/A" and cited_details["abstract"] == "N/A" and not cited_details["claims"]:
                        overall_cited_detail_lookup_failures += 1
                        error_detail=f"Essential details (title/abstract/claims) missing in record for cleaned num {cleaned_cited_num} (Raw: {agg_data['raw_nums']})"
                        overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Skipped Citation: Missing Details', error_detail)
                        citation_skipped = True
                        continue # Skip this citation entirely

                    spec_file_path = find_spec_file(spec_dir, cleaned_cited_num)
                    filtered_paragraphs = []

                    if not spec_file_path:
                        spec_lookup_failed_this_patent = True
                        error_detail=f"Spec file not found for cleaned num {cleaned_cited_num} (Raw: {agg_data['raw_nums']})"
                        overall_spec_lookup_failures += 1
                        overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Skipped Citation: Spec Not Found', error_detail)
                        citation_skipped = True
                        continue # Skip this citation entirely
                    else:
                        try:
                            with open(spec_file_path, 'r', encoding='utf-8') as sf: spec_data = json.load(sf)
                            all_paragraphs = spec_data.get("items", [])
                            if not isinstance(all_paragraphs, list): raise ValueError("'items' not a list in spec")

                            target_keys = agg_data["cited_texts"]
                            for item in all_paragraphs:
                                if isinstance(item, dict):
                                    parsed_key = parse_paragraph_key(item.get("key"))
                                    if parsed_key is not None and parsed_key in target_keys:
                                        filtered_paragraphs.append({
                                            "key": parsed_key,
                                            "content": item.get("content", "")
                                        })
                            if not filtered_paragraphs and target_keys:
                                 warn_detail = f"Spec file {spec_file_path.name} loaded, but no paragraphs found matching target keys {target_keys} for {cleaned_cited_num}"
                                 overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Warning: No Matching Paragraphs', warn_detail)

                        except Exception as e_spec:
                            spec_lookup_failed_this_patent = True
                            error_detail=f"Error processing spec file {spec_file_path.name} for {cleaned_cited_num}: {type(e_spec).__name__}: {e_spec}"
                            overall_spec_lookup_failures += 1
                            overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Skipped Citation: Spec Error', error_detail)
                            citation_skipped = True
                            continue # Skip this citation entirely

                    if not citation_skipped:
                        prior_art_entry = {
                            "patent_id": list(agg_data["raw_nums"])[0] if agg_data["raw_nums"] else "UNKNOWN",
                            "title": cited_details["title"],
                            "abstract": cited_details["abstract"],
                            "claims": cited_details["claims"],
                            "paragraphs": filtered_paragraphs # Contains paragraphs only if found and processed
                        }
                        prior_art_specifications_list.append(prior_art_entry)

                for claim_info_pc in claims_in_pc:
                    claim_number = None # Reset for each claim loop
                    try:
                        if not isinstance(claim_info_pc, dict): continue

                        claim_number = claim_info_pc.get("claimNumber")
                        if claim_number is None:
                            overall_claims_skipped_missing_data += 1
                            overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, None, 'Skipped Claim: No Number', f"Skipping claim entry in {pc_file_path.name} due to missing 'claimNumber'")
                            continue # Skip this claim entry

                        overall_claims_processed += 1
                        is_reject = claim_info_pc.get("isReject", False)
                        reasons = claim_info_pc.get("reasons", [])
                        answer_code = 'ALLOW'
                        answer_reason = "The claim is allowable over the prior art of record; the cited references do not teach or render obvious the present claims."
                        missing_reason_text_warning = False

                        if is_reject:
                            if isinstance(reasons, list) and reasons:
                                all_reason_texts = []
                                primary_code_found = False
                                for reason in reasons:
                                    if isinstance(reason, dict):
                                        if not primary_code_found:
                                            code = reason.get("sectionCode")
                                            if code:
                                                answer_code = str(code)
                                                primary_code_found = True
                                        reason_text = reason.get("reason")
                                        if isinstance(reason_text, str):
                                            all_reason_texts.append(reason_text.strip())
                                if not primary_code_found:
                                    answer_code = "REJECT_UNKNOWN"
                                answer_reason = "\n".join(all_reason_texts) if all_reason_texts else "Rejection reasons not specified."
                                if not all_reason_texts:
                                     missing_reason_text_warning = True
                            else: # isReject is True, but reasons list is missing or invalid
                                answer_code = "REJECT_UNKNOWN"
                                answer_reason = "Claim rejected but reasons are missing or invalid in source data."
                                missing_reason_text_warning = True

                        if is_reject and not prior_art_specifications_list:
                            warn_detail = f"Creating rejection benchmark (Code: {answer_code}) for claim {claim_number}, but no valid prior art specifications could be processed/included."
                            overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, claim_number, 'Warning: Rejection w/o Prior Art', warn_detail)
                        if missing_reason_text_warning:
                            warn_detail = f"Rejection benchmark (Code: {answer_code}) for claim {claim_number} has missing or default reason text: '{answer_reason}'"
                            overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, claim_number, 'Warning: Missing Reason Text', warn_detail)


                        benchmark_instance = {
                            "application_number": str(target_patent_number),
                            "claim_number": claim_number,
                            "context": {
                                "title": target_title,
                                "abstract": target_abstract,
                                "claims": initial_claims
                            },
                            "prior_art_specifications": prior_art_specifications_list, # Use the potentially filtered list
                            "answer": {
                                "reason": answer_reason,
                                "code": answer_code
                            }
                        }


                        indiv_save_failed_this_claim = False
                        try:
                            if not benchmark_instance["application_number"] or benchmark_instance["claim_number"] is None:
                                 raise ValueError("Cannot save benchmark instance due to missing application or claim number.")

                            individual_filename = f"rejection_{r_index}_{target_patent_number}_cl{claim_number}.json"
                            individual_filepath = individual_json_dir / individual_filename
                            with open(individual_filepath, 'w', encoding='utf-8') as f_indiv:
                                json.dump(benchmark_instance, f_indiv, ensure_ascii=False, indent=2)
                            overall_individual_files_saved += 1
                            overall_benchmark_data.append(benchmark_instance)
                            overall_benchmarks_created += 1
                            processed_claims_this_record += 1

                        except Exception as e_indiv:
                            indiv_save_failed_this_claim = True
                            overall_individual_files_failed += 1
                            error_detail = f"Error saving individual JSON {individual_filename}: {type(e_indiv).__name__}: {e_indiv}"
                            overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, claim_number, 'Error: Saving Individual JSON', error_detail)


                    except Exception as e_claim_proc:
                         overall_claims_skipped_missing_data += 1 # Count as skipped due to error
                         error_detail = f"Error processing claim {claim_number}: {type(e_claim_proc).__name__}: {e_claim_proc}"
                         overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name, claim_number, 'Error: Processing Claim', error_detail)

            except Exception as e_outer:
                error_detail = f"Outer Processing Error for {record_file.name} / {pc_file_path.name if pc_file_path else 'N/A'}: {type(e_outer).__name__}: {e_outer}\n{traceback.format_exc(limit=2)}"
                overall_log_entries_written += log_status(log_csv_writer, log_csv_file_handle, datetime.now().isoformat(), record_file.name, pc_file_path.name if pc_file_path else None, None, 'Error: Processing Record/pC (Outer)', error_detail)
                print(f"  ❌ ERROR: {error_detail}. Skipping rest of record.", file=sys.stderr)

            status_emoji = "❌ FAILED" if (record_load_error or pc_load_error) else ("⚠️ PARTIAL" if processed_claims_this_record == 0 and overall_claims_processed > 0 else "✅ COMPLETED") # Adjust logic slightly
            print(f"--- [{i+1}/{total_records}] Finished Record: {record_file.name} ({status_emoji}) - Benchmarks Created: {processed_claims_this_record} ---")

    finally:
        if log_csv_file_handle:
            log_csv_file_handle.close()
            print(f"\nDetailed log file closed: {detailed_log_path}")

    print(f"\n--- Overall Rejection Benchmark Generation Summary ---")
    print(f"Total Record files found: {total_records}")
    print(f"Record files successfully processed (Record & pC loaded): {overall_records_processed}")
    if overall_records_failed_loading > 0: print(f"Record files failed to load: {overall_records_failed_loading}")
    if overall_pc_files_not_found > 0: print(f"Matching pC files not found: {overall_pc_files_not_found}")
    if overall_pc_files_failed_loading > 0: print(f"pC files failed to load/parse: {overall_pc_files_failed_loading}")
    print(f"Total Claims processed across all files: {overall_claims_processed}")
    if overall_claims_skipped_missing_data > 0: print(f"Claims skipped due to errors or missing essential data: {overall_claims_skipped_missing_data}")
    if overall_cited_detail_lookup_failures > 0: print(f"Cited Patent detail lookups failed (skipped citations): {overall_cited_detail_lookup_failures}")
    if overall_spec_lookup_failures > 0: print(f"Specification file lookup/processing failures (skipped citations): {overall_spec_lookup_failures}")
    print(f"Total benchmark instances generated (and added to aggregated list): {overall_benchmarks_created}")
    print(f"Total individual JSON files successfully saved: {overall_individual_files_saved}")
    if overall_individual_files_failed > 0: print(f"Errors saving individual JSON files: {overall_individual_files_failed}")
    print(f"Total log entries written to {detailed_log_path}: {overall_log_entries_written}")

    if overall_benchmark_data:
        print(f"\nSaving {len(overall_benchmark_data)} benchmark instances to {output_file_jsonl}...")
        try:
            with open(output_file_jsonl, 'w', encoding='utf-8') as f_out:
                for entry in overall_benchmark_data:
                    json.dump(entry, f_out, ensure_ascii=False); f_out.write('\\n')
            print("Benchmark data saved successfully to JSONL.")
        except Exception as e_jsonl:
             print(f"Error writing benchmark JSONL file {output_file_jsonl}: {e_jsonl}", file=sys.stderr)
    else:
        print("\nNo benchmark data generated for aggregated files (JSONL/Parquet). Check logs for skipped items.")


    if overall_benchmark_data:
        print(f"Saving {len(overall_benchmark_data)} benchmark instances to {output_file_parquet}...")
        try:
            df_benchmark = pd.DataFrame(overall_benchmark_data)
            for col in ['context', 'prior_art_specifications', 'answer']:
                if col in df_benchmark.columns:
                     needs_serialization = df_benchmark[col].apply(lambda x: isinstance(x, (dict, list))).any()
                     if needs_serialization:
                         try:
                             df_benchmark[col] = df_benchmark[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
                         except Exception as e_serial:
                             print(f"Warning: Could not serialize column '{col}' to JSON string for Parquet: {e_serial}. Attempting direct save.", file=sys.stderr)

            df_benchmark.to_parquet(output_file_parquet, index=False)
            print("Benchmark data saved successfully to Parquet.")
        except ImportError:
            print("Error: 'pandas' and 'pyarrow' are required to save to Parquet. Please install them (`pip install pandas pyarrow`).", file=sys.stderr)
        except Exception as e_parquet:
            print(f"Error writing benchmark Parquet file {output_file_parquet}: {e_parquet}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

# --- Entry Point ---
if __name__ == "__main__":
    try:
        import pandas
        import pyarrow
    except ImportError:
        print("FATAL ERROR: 'pandas' and 'pyarrow' libraries are required for Parquet output.", file=sys.stderr)
        print("Please install them using: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    try:
        script_dir = Path(__file__).resolve().parent
        base_data_dir = script_dir.parent / "data"
        if not base_data_dir.is_dir():
             print(f"FATAL ERROR: Base data directory not found at expected location: {base_data_dir}", file=sys.stderr)
             sys.exit(1)

        record_input_dir = base_data_dir / 'record'
        pc_input_dir = base_data_dir / 'parsed_CTNF_with_PN'
        spec_input_dir = base_data_dir / 'spec_cited' / 'text' / 'parsed'
        output_jsonl_file = base_data_dir / 'benchmarks' / 'noc4pc_benchmark.jsonl'
        output_parquet_file = base_data_dir / 'benchmarks' / 'noc4pc_benchmark.parquet'
        individual_json_output_dir = base_data_dir / 'benchmarks' / 'noc4pc' # New dir for individual files
        error_report_dir = base_data_dir / 'error_report' # Reusing error report dir

        create_rejection_benchmark(
            record_dir=record_input_dir,
            pc_dir=pc_input_dir,
            spec_dir=spec_input_dir,
            output_file_jsonl=output_jsonl_file,
            output_file_parquet=output_parquet_file,
            individual_json_dir=individual_json_output_dir,
            error_dir=error_report_dir
        )
        print("\n--- Script Finished ---")
    except Exception as main_e:
        print(f"\nFATAL SCRIPT ERROR: {type(main_e).__name__}: {main_e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


