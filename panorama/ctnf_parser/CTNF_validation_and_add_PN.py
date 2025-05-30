from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import json
import os
import re
import csv
from datetime import datetime
from pathlib import Path
import tiktoken

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(model="gpt-4o")

current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = Path("./data/parsed_CTNF")
output_dir = Path("./data/parsed_CTNF_valid")
validation_dir = Path("./data/validation")
error_dir = Path("./data/error_report")
PROMPT_PATH = os.path.join(current_dir, "paragraphNum_prompt.txt")

def load_prompt():
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found`: {PROMPT_PATH}")
    
def extract_int_from_pC_filename(filename: str) -> int:
    """
    Given a filename like 'pC_r00017_13793810.json', extract the integer part after 'pC_r' and
    before the next underscore. For example, '00017' -> 17.
    If no match is found, return a large number or 0 as default.
    """
    match = re.search(r"pC_r(\d+)_", filename)
    if match:
        return int(match.group(1))
    return 999999  # Or 0, if you prefer to push unrecognized files to the end.

def extract_int_from_rec_filename(filename: str) -> int:
    """
    Given a filename like 'rec_r00017_13793810.json', extract the integer part after 'rec_r'
    and before the next underscore. For example, '00017' -> 17.
    Returns None or a default if not found.
    """
    match = re.search(r"rec_r(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None

def filter_102_103_claims(ctnf_data):
    filtered_claims = []
    has_valid_rejection = False
    
    for claim in ctnf_data["claims"]:
        valid_reasons = []
        for reason in claim.get("reasons", []):
            if reason.get("sectionCode") in [102, 103]:
                valid_reasons.append(reason)
                has_valid_rejection = True
        
        claim["reasons"] = valid_reasons
        if not valid_reasons:
            claim["isReject"] = False
        filtered_claims.append(claim)
    
    ctnf_data["claims"] = filtered_claims
    return ctnf_data, has_valid_rejection

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)

def validate_and_copy_ctnf():
    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    # Setup error logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_log_path = error_dir / f"error_ctnf_validation_{current_time}.csv"

    with open(error_log_path, 'w', newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["app_num", "error_code", "error_message"])
        writer.writeheader()

    # Read validation results
    validation_result_path = validation_dir / "validation_result.csv"
    valid_apps = []
    
    with open(validation_result_path, 'r', newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["valid_b"].lower() == "true":
                valid_apps.append(row["app_num"])

    print(f"Total valid applications: {len(valid_apps)}")

    successful_count = 0
    for app_num in valid_apps:
        try:
            ctnf_files = list(input_dir.glob(f"pC_*_{app_num}.json"))

            if not ctnf_files or len(ctnf_files) > 1:
                print(f"Warning: Invalid CTNF files for {app_num}")
                continue

            ctnf_file = ctnf_files[0]

            with open(ctnf_file, 'r') as f:
                ctnf_data = json.load(f)

            filtered_ctnf, has_valid_rejection = filter_102_103_claims(ctnf_data)
            
            if not has_valid_rejection:
                with open(error_log_path, 'a', newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["app_num", "error_code", "error_message"])
                    writer.writerow({
                        "app_num": app_num,
                        "error_code": 300,
                        "error_message": "No 102/103 rejections found"
                    })
                print(f"❌ No 102/103 rejections found for application {app_num}")
                continue

            output_file = output_dir / ctnf_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_ctnf, f, indent=4, ensure_ascii=False)

            successful_count += 1
            print(f"✅ Validated and copied CTNF for application {app_num}")

            with open(error_log_path, 'a', newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["app_num", "error_code", "error_message"])
                writer.writerow({
                    "app_num": app_num,
                    "error_code": -1,
                    "error_message": ""
                })

        except Exception as e:
            print(f"Error processing application {app_num}: {e}")
            with open(error_log_path, 'a', newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["app_num", "error_code", "error_message"])
                writer.writerow({
                    "app_num": app_num,
                    "error_code": 400,
                    "error_message": str(e)
                })

    print(f"Successfully processed {successful_count} out of {len(valid_apps)} applications")

def add_paragraph_num():
    input_dir = Path("./data/parsed_CTNF_valid")
    output_dir = Path("./data/parsed_CTNF_with_PN")
    output_dir.mkdir(parents=True, exist_ok=True)

    error_dir = Path("./data/error_report")
    error_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_log_path = error_dir / f"error_paragraph_num_{current_time}.csv"

    with open(error_log_path, 'w', newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "error_message"])
        writer.writeheader()

    gpt_prompt = load_prompt()

    # 1) Get all pC_r files
    json_files = list(input_dir.glob("*.json"))
    # 2) Sort them by integer extracted from pC_rXXXXX
    json_files.sort(key=lambda p: extract_int_from_pC_filename(p.name))

    print(f"Found {len(json_files)} files in {input_dir} (sorted by rXXXXX integer).")

    # Prepare data/record folder for matching rec_rXXXXX
    record_dir = Path("./data/record")
    record_files = list(record_dir.glob("*.json"))

    success_count = 0

    for json_path in json_files:
        file_name = json_path.name
        try:
            # Extract integer from "pC_rXXXXX"
            pC_int_val = extract_int_from_pC_filename(file_name)
            if pC_int_val < 9617:
                continue

            # Read original data
            with open(json_path, "r", encoding="utf-8") as rf:
                original_data = json.load(rf)

            # 3) Find matching rec_r file in data/record
            matched_body_text = ""
            candidate_files = []
            for rf_path in record_files:
                rec_int_val = extract_int_from_rec_filename(rf_path.name)
                if rec_int_val == pC_int_val:
                    candidate_files.append(rf_path)

            if len(candidate_files) >= 1:
                with open(candidate_files[0], "r", encoding="utf-8") as rec_f:
                    rec_data = json.load(rec_f)
                    matched_body_text = rec_data.get("CTNFBodyText", "")

            # Check combined token count using tiktoken
            content_json_str = json.dumps(original_data, ensure_ascii=False)
            total_tokens = count_tokens(content_json_str) + count_tokens(matched_body_text)
            if total_tokens > 29000:
                msg = f"Combined token count {total_tokens} exceeds 29000, skipping GPT."
                print(f"❌ {msg} -> {file_name}")
                with open(error_log_path, 'a', newline="", encoding='utf-8') as ef:
                    writer = csv.DictWriter(ef, fieldnames=["file_name", "error_message"])
                    writer.writerow({"file_name": file_name, "error_message": msg})
                continue

            # 4) Send to GPT
            messages = [
                HumanMessage(
                    content=(
                        gpt_prompt
                        + "\n\n---\nHere is the JSON to transform:\n\n"
                        + content_json_str
                        + "\n\n---\nHere is the original CTNF Document:\n\n"
                        + matched_body_text
                    )
                )
            ]

            response = chat(messages)
            gpt_output = response.content.strip()

            # 5) Parse GPT's output as JSON
            if gpt_output.startswith("```json") and gpt_output.endswith("```"):
                gpt_output = gpt_output[7:-3].strip()

            try:
                transformed_data = json.loads(gpt_output)
            except json.JSONDecodeError as e:
                raise ValueError(f"Could not parse GPT response as JSON. Error: {e}")

            # 6) Save the transformed JSON
            output_path = output_dir / file_name
            with open(output_path, "w", encoding="utf-8") as wf:
                json.dump(transformed_data, wf, indent=4, ensure_ascii=False)

            success_count += 1
            print(f"✅ Processed {file_name}")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            with open(error_log_path, 'a', newline="", encoding='utf-8') as ef:
                writer = csv.DictWriter(ef, fieldnames=["file_name", "error_message"])
                writer.writerow({"file_name": file_name, "error_message": str(e)})

    print(f"Completed. Total files: {len(json_files)}, Success: {success_count}, Failed: {len(json_files) - success_count}")

if __name__ == "__main__":
    validate_and_copy_ctnf()
    add_paragraph_num()
