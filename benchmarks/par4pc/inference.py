# example(zero-shot): python benchmarks/par4pc/inference.py --provider openai --model gpt-4o --prompt_mode zero-shot
# example(cot): python benchmarks/par4pc/inference.py --provider openai --model gpt-4o --prompt_mode cot
# example(baseline): python benchmarks/par4pc/inference.py --baseline --runs 10
# example(baseline-2): python benchmarks/par4pc/inference.py --baseline-2 --runs 10

import json
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime
from dotenv import load_dotenv
import traceback
import time
import re
import numpy as np
import warnings
import csv
import random
from pydantic import BaseModel, Field
from datasets import load_dataset


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

warnings.filterwarnings(action='ignore')


class ZeroShotAnswer(BaseModel):
    answer: Union[str, List[str]] = Field(description="The letter(s) of the cited patent(s).")

class CoTAnswer(BaseModel):
    reason: str = Field(description="Step-by-step reasoning for the answer (max 200 words).")
    answer: Union[str, List[str]] = Field(description="The letter(s) of the cited patent(s) based on the reasoning.")


def get_chat_model(provider: str, model_name: str, prompt_mode:str):
    if provider.lower() == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
    elif provider.lower() == 'anthropic':
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        need_reason = (prompt_mode != "zero-shot")
        required_keys = ["answer", "reason"] if need_reason else ["answer"]
        anthropic_tools = [{
            "name": "select_paragraph",
            "description": "Return the paragraph key (and optional reasoning)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": required_keys
            }
        }]

        return ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            tools=anthropic_tools,
            tool_choice={"type": "tool", "name": "select_paragraph"}
        )
    elif provider.lower() == 'google':
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        chat_model_base = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            convert_system_message_to_human=True,
            model_kwargs={
                "generation_config": {
                    "response_mime_type": "application/json",
                    "temperature": 0
                }
            }
        )

        if prompt_mode == "cot" or "cot_base":
            structured_llm = chat_model_base.with_structured_output(CoTAnswer)
        else: # zero-shot
            structured_llm = chat_model_base.with_structured_output(ZeroShotAnswer)

        return structured_llm

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_par4pc_prompt(par4pc_data: Dict[str, Any], prompt_mode: str) -> str:
    context = par4pc_data.get("context", {})
    options = par4pc_data.get("options", {})
    claim_number = par4pc_data.get("claim_number", "N/A")
    app_number = par4pc_data.get("application_number", "N/A")

    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode context JSON for app {app_number}, claim {claim_number}. Using raw string.")
            context = {"raw_string": context}
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode options JSON for app {app_number}, claim {claim_number}. Using empty dict.")
            options = {}

    context_claims_data = context.get("claims", [])
    if isinstance(context_claims_data, np.ndarray):
        context_claims_data = context_claims_data.tolist()
    context_claims_json = json.dumps(context_claims_data, indent=4)

    prompt = f"""
You are a patent expert tasked with identifying cited patents for a specific claim rejection based *only* on the provided context and options.

**Context:**
*   **Application Number:** {app_number}
*   **Title:** {context.get("title", "N/A")}
*   **Abstract:** {context.get("abstract", "N/A")}
*   **Initial Claims:**
    {context_claims_json}

**Target Claim for Analysis:** Claim {claim_number}

**Options (Potential Cited Patents):**
"""
    for letter, details in sorted(options.items()):
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode details JSON for option {letter} in app {app_number}, claim {claim_number}.")
                details = {}
        elif not isinstance(details, dict):
            print(f"Warning: Option {letter} details is not a dictionary (type: {type(details)}). Treating as empty.")
            details = {}

        prompt += f"\n{letter}: Patent ID: {details.get('patent_id', 'N/A')}\n"
        prompt += f"   Title: {details.get('title', 'N/A')}\n"
        prompt += f"   Abstract: {details.get('abstract', 'N/A')}\n"

        option_claims_data = details.get('claims', [])
        if isinstance(option_claims_data, np.ndarray):
            option_claims_data = option_claims_data.tolist()
        claims_str = json.dumps(option_claims_data, indent=4) if option_claims_data else "[]"

        prompt += f"   Claims: {claims_str}\n"

    if prompt_mode == "cot":
        prompt += f"""
**Step-by-Step Instrcution**  
*Apply the Broadest Reasonable Interpretation (BRI) standard.*

1. **BRI Claim Charting**  
   • Decompose Claim {claim_number} into numbered limitations [L1]-[Ln] and record element : function/relationship.

2. **Core Inventive Concept & Problem**  
   • Summarise in ≤ 20 words the inventive concept + technical problem.

3. **Single-Reference Screening (§102)**  
   • For each option (A-H) rate coverage:  
     | Opt | Maps limits | Term/synonym | Field match | Score* |  
     |-----|------------|---------------|-------------|--------|  
     *Score: 0 = no key feature, 1 = partial, 2 = full anticipation.*

4. **Multi-Reference Analysis (§103)**  
   a. Pick options with Score ≥ 1.  
   b. Build coverage matrix to find smallest combo covering all limits.  
   c. For each viable combo, supply a motivation-to-combine (same field, complementary function, predictable substitution, etc.).  
   d. Rank: full coverage → clear motivation → earliest primary art.

5. **Consistency & Inherency Check**  
   • Reject art that contradicts any limitation; accept inherent feature only if necessarily present.

6. **Output (JSON only)**  
    Always write the "reason" **first** and then write the "answer".
   • "reason" MUST include:  
     Step1 <claim focus>; Step2 <mapping & motivation> ; Step3 <§102 or §103>.  
   • Keep "reason" ≤ 200 words.
   • "answer" = single letter **or** list of letters.

```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ...", "answer":"A"}}] 
```
If multiple patents are cited":
```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ...", "answer": ["A","C","F"]}}]
```
""" 
    if prompt_mode == "cot_base":
        prompt += f"""
Based only on the provided context and options, which patent(s) (A-H) were cited?
Think through the steps required to evaluate this, craft the supporting rationale accordingly, and then deliver your answer based on that rationale.
Always write the "reason" **first** and then write the "answer".

Answer format (JSON only):
```json
{{"reason":"", "answer": "A"}}
```
If multiple patents are cited
```json
{{"reason":"", "answer": ["A","C","F"]}}
```
"""        
    
    else: # ---------- zero‑shot ---------- 
        prompt += f"""
Based only on the provided context and options, which patent(s) (A-H) were cited?
Answer format (JSON only):
```json
{{"answer": "A"}}
```
If multiple patents are cited
```json
{{"answer": ["A","C","F"]}}
```
"""
    return prompt


def process_llm_response(response_data: Union[dict, BaseModel]) -> List[str]:
    ans = []
    if isinstance(response_data, BaseModel):
        try:
            ans = response_data.answer
        except AttributeError:
            print(f"Warning: 'answer' attribute not found in Pydantic model: {response_data}")
            ans = []
    elif isinstance(response_data, dict):
        ans = response_data.get("answer", "")
        if isinstance(ans, str):
            ans = list(filter(lambda c: 'A' <= c <= 'H', re.split(r'[,\s]+', ans.upper())))

    else:
        print(f"Warning: Unexpected response type for process_llm_response: {type(response_data)}. Trying direct parse.")
        try:
            response_content = str(response_data)
            data = json.loads(response_content)
            ans = data.get("answer", [])
        except Exception:
            print(f"Error: Could not process response: {response_data}")
            return []

    if isinstance(ans, str):
        if len(ans.strip()) == 1 and 'A' <= ans.strip().upper() <= 'H':
             ans = [ans.strip().upper()]
        else:
            ans = list(filter(lambda c: 'A' <= c <= 'H', re.split(r'[,\s]+', ans.upper())))

    if isinstance(ans, list):
        processed_ans = []
        for item in ans:
            if isinstance(item, str) and len(item.strip()) == 1 and 'A' <= item.strip().upper() <= 'H':
                processed_ans.append(item.strip().upper())
        return sorted(list(set(processed_ans)))
    else:
        print(f"Warning: Could not normalize 'answer' to a list: {ans}")
        return []


def process_answer_string(answer_string: str | np.ndarray) -> List[str]:

    if isinstance(answer_string, np.ndarray):
        answer_string = ','.join(str(x) for x in answer_string)

    return sorted(list(set(filter(lambda c: 'A' <= c <= 'H', re.split(r'[,\s]+', answer_string.upper())))))


def evaluate_prediction(predicted_letters: List[str], gold_answers: List[str]) -> bool:
    if isinstance(gold_answers, str):
        gold_answers = process_answer_string(gold_answers)

    if isinstance(predicted_letters, str):
        predicted_letters = process_answer_string(predicted_letters)
    
    return set(predicted_letters) == set(gold_answers)

def format_answer_list_for_csv(answer_data):
    letters = []

    if isinstance(answer_data, str):
        stripped = answer_data.strip()

        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped).get("answer", [])
                if isinstance(parsed, str):
                    parsed = [parsed]
                answer_data = parsed
            except Exception:
                stripped = stripped

        if not isinstance(answer_data, (list, np.ndarray)):
            letters = sorted(list(set(
                filter(lambda c: 'A' <= c <= 'Z', re.split(r'[,\s]+', stripped.upper()))
            )))

    if isinstance(answer_data, (list, np.ndarray)):
        if isinstance(answer_data, np.ndarray):
            answer_data = answer_data.tolist()
        processed = [
            str(item).strip().upper()
            for item in answer_data
            if isinstance(item, (str, int)) and len(str(item).strip()) == 1
        ]
        letters = sorted(list(set(
            [c for c in processed if 'A' <= c <= 'Z']
        )))

    if not letters and answer_data not in (None, [], {}):
        try:
            s = str(answer_data).strip().upper()
            if len(s) == 1 and 'A' <= s <= 'Z':
                letters = [s]
        except Exception:
            pass

    return ",".join(letters)


def calculate_custom_score(predicted_letters: List[str], gold_letters: List[str], silver_letters: List[str]):
    pred_set = set(predicted_letters)
    gold_set = set(gold_letters)
    silver_set = set(silver_letters)

    max_score = len(gold_set) * 2 # max score

    model_score = 0

    tp_g = pred_set.intersection(gold_set)
    fp = pred_set - gold_set - silver_set # Predicted but not gold or silver
    fn_g = gold_set - pred_set # Gold but not predicted

    model_score += len(tp_g) * 2  # Score for correctly predicted Gold
    model_score -= len(fp) * 1    # Penalty for False Positives
    model_score -= len(fn_g) * 1  # Penalty for missed Gold (FN Gold)

    raw_score = model_score
    model_score = max(0, model_score)

    normalized_score = 0
    if max_score > 0:
        normalized_score = (model_score / max_score) * 100 if max_score > 0 else 0
    
    return normalized_score, 100

def calculate_3x2_matrix(
    gold_answers: List[str],
    silver_answers: List[str],
    negative_answers: List[str],
    predicted_answers: List[str]
) -> dict:
    """Return a 3x2 confusion-matrix count dictionary."""
    gold_set     = set(gold_answers)
    silver_set   = set(silver_answers)
    negative_set = set(negative_answers)
    pred_set     = set(predicted_answers)

    return {
        # TP
        "gold_answer"    : len(pred_set & gold_set),
        "silver_answer"  : len(pred_set & silver_set),
        "negative_answer": len(pred_set & negative_set),
        # FN / TN
        "gold_negative"     : len(gold_set     - pred_set),
        "silver_negative"   : len(silver_set   - pred_set),
        "negative_negative" : len(negative_set - pred_set),
    }


def evaluate_and_log_3x2_matrix(gold_answers: List[str], silver_answers: List[str], negative_answers: List[str], predicted_answers: List[str], result_row_dict: dict):
    matrix = calculate_3x2_matrix(gold_answers, silver_answers, negative_answers, predicted_answers)
    
    result_row_dict["gold_answer"]      = matrix["gold_answer"]
    result_row_dict["silver_answer"]    = matrix["silver_answer"]
    result_row_dict["negative_answer"]  = matrix["negative_answer"]

    result_row_dict["gold_negative"]    = matrix["gold_negative"]
    result_row_dict["silver_negative"]  = matrix["silver_negative"] 
    result_row_dict["negative_negative"]= matrix["negative_negative"]


DATASET_NAME = "DxD-Lab/PANORAMA-PAR4PC-Bench"

def main(provider: str, model_name: str, prompt_mode: str):
    print(f"Starting benchmark test with {provider} - {model_name} using Hugging Face dataset {DATASET_NAME} (test split)")
    
    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        total_par4pcs = len(df)
        print(f"Loaded {total_par4pcs} par4pcs from Hugging Face dataset {DATASET_NAME} (test split).")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return

    if df.empty:
        print(f"Error: No data found in the dataset.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir_name = f'result_{timestamp}_{provider}_{model_name}_{prompt_mode}_{dataset_name_for_path}'
        
    result_dir = Path(__file__).parent / 'result' / result_dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    results_file = result_dir / "evaluation_results.csv"
    error_log_file = result_dir / "error_log.txt"

    csv_header = ["identifier", "claim_number", "application_number",
                  "gold_answers", "silver_answers", "negative_answers",
                  "predicted_answers", "llm_raw_response", "is_correct", "error",
                  "model_score", "max_score", 
                  "gold_answer", "silver_answer", "negative_answer",
                  "gold_negative", "silver_negative", "negative_negative"]

    try:
        with open(results_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
        print(f"Results CSV file initialized: {results_file}")
    except IOError as e:
        print(f"Error initializing CSV file {results_file}: {e}")
        return

    correct_predictions = 0
    errors = 0
    total_model_score = 0
    total_max_score = 0

    all_gold_answers = []
    all_silver_answers = []
    all_negative_answers = []
    all_predicted_answers = []

    total_gold_answer = 0
    total_silver_answer = 0
    total_negative_answer = 0
    total_gold_negative = 0
    total_silver_negative = 0
    total_negative_negative = 0

    chat = get_chat_model(provider, model_name, prompt_mode)

    for index, row in df.iterrows():
        print(f"\n\rProcessing: {index + 1}/{total_par4pcs} ({((index + 1)/total_par4pcs)*100:.1f}%)", end="")

        par4pc_identifier = f"app_{row.get('application_number', 'N/A')}_claim_{row.get('claim_number', 'N/A')}"

        llm_response_object = None
        llm_raw_response_str = ""
        predicted_letters = []
        is_correct = False
        error_message = ""
        model_score = 0
        max_score = 0

        result_row_dict = {
            "identifier": par4pc_identifier,
            "claim_number": row.get("claim_number", "N/A"),
            "application_number": row.get("application_number", "N/A"),
            "gold_answers": "", "silver_answers": "", "negative_answers": "",
            "predicted_answers": "", "llm_raw_response": "",
            "is_correct": False, "error": "",
            "model_score": 0, "max_score": 0,
            "gold_answer": 0, "silver_answer": 0, "negative_answer": 0,
            "gold_negative": 0, "silver_negative": 0, "negative_negative": 0
        }

        try:
            par4pc_data_dict = row.to_dict()
            prompt = create_par4pc_prompt(par4pc_data_dict, prompt_mode)

            gold_answers_raw = row.get("gold_answers")
            gold_answers_list = process_answer_string(gold_answers_raw)
            all_gold_answers.append(gold_answers_list)

            silver_answers_raw = row.get("silver_answers")
            silver_answers_list = process_answer_string(silver_answers_raw)
            all_silver_answers.append(silver_answers_list)

            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    system_message = SystemMessage(content="You are an expert patent examiner identifying cited patents based on context and options. Respond in the requested JSON format.")
                    human_message = HumanMessage(content=prompt)

                    if provider.lower() == 'google':
                        llm_response_object = chat.invoke([system_message, human_message])

                        if llm_response_object is None:
                            raise ValueError("LLM invoke returned None. Possible API issue or structured output failure.")

                        llm_raw_response_str = llm_response_object.model_dump_json()

                    elif provider.lower() == "anthropic":
                        response = chat([system_message, human_message])
                        if response.tool_calls:
                            llm_response_object = response.tool_calls[0]["args"]
                            llm_raw_response_str = json.dumps(llm_response_object)
                        else:
                            llm_response_object = {"answer": ""}
                            llm_raw_response_str = response.content.strip()
                            print(f"\nWarning: Anthropic response did not contain tool calls: {llm_raw_response_str}")

                    else:
                         response = chat([system_message, human_message])
                         llm_raw_response_str = response.content.strip()
                         try:
                             llm_response_object = json.loads(llm_raw_response_str)
                         except json.JSONDecodeError:
                             print(f"\nWarning: Could not parse OpenAI response as JSON: {llm_raw_response_str}")
                             llm_response_object = {"answer": []}

                    predicted_letters = process_llm_response(llm_response_object)
                    error_message = ""
                    success = True
                    break
                except Exception as e:
                    error_message = f"LLM call failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}"
                    llm_raw_response_str = f"ERROR: {error_message}"
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)


            if not success:
                errors += 1

            if success:
                 is_correct = evaluate_prediction(predicted_letters, gold_answers_list)
                 if is_correct:
                     correct_predictions += 1

                 model_score, max_score = calculate_custom_score(predicted_letters, gold_answers_list, silver_answers_list)
                 total_model_score += model_score
                 total_max_score += max_score
            else:
                 try:
                     _, max_score_on_error = calculate_custom_score([], gold_answers_list, silver_answers_list)
                     max_score = max_score_on_error
                     total_max_score += max_score_on_error
                 except Exception as score_err:
                     print(f"Error calculating max score on error: {score_err}")
                     max_score = 0


            negative_answers_raw  = row.get("negative_answers")
            negative_answers_list = process_answer_string(negative_answers_raw)
            evaluate_and_log_3x2_matrix(
                gold_answers_list,
                silver_answers_list,
                negative_answers_list,
                predicted_letters,
                result_row_dict
            )

            total_gold_answer += result_row_dict["gold_answer"]
            total_silver_answer += result_row_dict["silver_answer"]
            total_negative_answer += result_row_dict["negative_answer"]
            total_gold_negative += result_row_dict["gold_negative"]
            total_silver_negative += result_row_dict["silver_negative"]
            total_negative_negative += result_row_dict["negative_negative"]

            result_row_dict["gold_answers"] = format_answer_list_for_csv(row.get("gold_answers"))
            result_row_dict["silver_answers"] = format_answer_list_for_csv(row.get("silver_answers"))
            result_row_dict["negative_answers"] = format_answer_list_for_csv(row.get("negative_answers"))
            result_row_dict["predicted_answers"] = ",".join(predicted_letters)
            result_row_dict["llm_raw_response"] = llm_raw_response_str
            result_row_dict["is_correct"] = is_correct if success else False
            result_row_dict["error"] = error_message if not success else ""
            result_row_dict["model_score"] = model_score
            result_row_dict["max_score"] = max_score

        except Exception as e:
            error_msg = f"Error processing row {index} ({par4pc_identifier}): {type(e).__name__}: {e}"
            print(f"\n  [!] Error processing row {index} ({par4pc_identifier}): {error_msg}")
            print(traceback.format_exc())
            errors += 1
            with open(error_log_file, 'a', encoding='utf-8') as f_err:
                 f_err.write(f"{datetime.now().isoformat()} - {error_msg}\n")
                 f_err.write(traceback.format_exc() + "\n\n")
            result_row_dict["error"] = error_msg
            try:
                gold_answers_list_on_error = process_answer_string(row.get("gold_answers"))
                silver_answers_list_on_error = process_answer_string(row.get("silver_answers"))
                _, max_score_on_error = calculate_custom_score([], gold_answers_list_on_error, silver_answers_list_on_error)
                result_row_dict["max_score"] = max_score_on_error
                total_max_score += max_score_on_error
            except Exception as score_err:
                print(f"Error calculating max score on outer error: {score_err}")
                result_row_dict["max_score"] = 0

        try:
            with open(results_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writerow(result_row_dict)
        except IOError as e:
            print(f"\n  [!] Error writing row {index} to CSV {results_file}: {e}")

    print(f"\n\n--- Final Results ---")
    print(f"Total par4pcs processed: {total_par4pcs}")
    print(f"Exact match count: {correct_predictions} ({correct_predictions/total_par4pcs*100:.2f}%)")
    
    if total_max_score > 0:
        average_score_percentage = (total_model_score / total_max_score) * 100
        print(f"Total custom score: {total_model_score:.2f} / {total_max_score} ({average_score_percentage:.2f}%) [max max]")
    else:
        print(f"Total custom score: {total_model_score:.2f} / {total_max_score} (Max score is zero, percentage not applicable)")
    
    print(f"Total errors during processing: {errors}")

    print(f"\nTotal Confusion Matrix (3x2):")
    print(f"               | Predicted Positive | Predicted Negative |")
    print(f"---------------|--------------------|--------------------|")
    print(f"Actual Gold    | {total_gold_answer:<18} | {total_gold_negative:<18} |")
    print(f"Actual Silver  | {total_silver_answer:<18} | {total_silver_negative:<18} |")
    print(f"Actual Negative| {total_negative_answer:<18} | {total_negative_negative:<18} |")

    print(f"\nDetailed results saved incrementally to: {results_file}")
    if os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 0:
         print(f"Error details logged to: {error_log_file}")
    elif errors > 0:
         print(f"Processing errors occurred, check terminal output for details.")


def run_baseline_evaluation(runs: int = 1):
    """
    Baseline: Randomly select 1+ options (A-H)
    runs: number of repetitions to compute statistics
    """
    print(f"Starting Baseline evaluation with Hugging Face dataset {DATASET_NAME} (test split)")
    print(f"Running {runs} times to compute statistics")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        total_par4pcs = len(df)
        print(f"Loaded {total_par4pcs} par4pcs from Hugging Face dataset {DATASET_NAME} (test split).")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return
    
    dataset_name_for_path = DATASET_NAME.replace("/", "_")

    if df.empty:
        print(f"Error: No data found in the dataset.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_dir_name = f'result_{timestamp}_baseline_runs{runs}_{dataset_name_for_path}'
    result_dir = Path(__file__).parent / 'result' / result_dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = result_dir / "individual_runs"
    runs_dir.mkdir(exist_ok=True)
    
    stats_file = result_dir / "baseline_stats.csv"
    error_log_file = result_dir / "error_log.txt"

    all_exact_match_rates = []
    all_custom_scores = []
    all_gold_answers = []
    all_silver_answers = []
    all_negative_answers = []
    all_confusion_matrices = []

    all_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    for run_idx in range(1, runs + 1):
        print(f"\n--- Starting Baseline Run {run_idx}/{runs} ---")
        
        run_result_file = runs_dir / f"run_{run_idx}_results.csv"
        
        csv_header = ["identifier", "claim_number", "application_number",
                    "gold_answers", "silver_answers", "negative_answers",
                    "predicted_answers", "baseline_selection_info", "is_correct", "error",
                    "model_score", "max_score", 
                    "gold_answer", "silver_answer", "negative_answer",
                    "gold_negative", "silver_negative", "negative_negative"]

        try:
            with open(run_result_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
            print(f"Run {run_idx} results CSV file initialized: {run_result_file}")
        except IOError as e:
            print(f"Error initializing CSV file {run_result_file}: {e}")
            continue

        correct_predictions = 0
        errors = 0
        total_model_score = 0
        total_max_score = 0

        run_confusion_totals = {
            "gold_answer": 0, "silver_answer": 0, "negative_answer": 0,
            "gold_negative": 0, "silver_negative": 0, "negative_negative": 0
        }

        for index, row in df.iterrows():
            print(f"\rProcessing Run {run_idx}/{runs}: {index + 1}/{total_par4pcs} ({((index + 1)/total_par4pcs)*100:.1f}%)", end="")

            par4pc_identifier = f"app_{row.get('application_number', 'N/A')}_claim_{row.get('claim_number', 'N/A')}"
        
            num_options = random.randint(1, len(all_options))
            predicted_letters = random.sample(all_options, num_options)
            predicted_letters.sort()
            baseline_selection_info = f"Randomly selected {num_options} option(s)"
            
            is_correct = False
            error_message = ""
            model_score = 0
            max_score = 0

            result_row_dict = {
                "identifier": par4pc_identifier,
                "claim_number": row.get("claim_number", "N/A"),
                "application_number": row.get("application_number", "N/A"),
                "gold_answers": "", "silver_answers": "", "negative_answers": "",
                "predicted_answers": "", "baseline_selection_info": baseline_selection_info,
                "is_correct": False, "error": "",
                "model_score": 0, "max_score": 0,
                "gold_answer": 0, "silver_answer": 0, "negative_answer": 0,
                "gold_negative": 0, "silver_negative": 0, "negative_negative": 0
            }

            try:
                gold_answers_raw = row.get("gold_answers")
                gold_answers_list = process_answer_string(gold_answers_raw)
                
                if run_idx == 1:
                    all_gold_answers.append(gold_answers_list)

                silver_answers_raw = row.get("silver_answers")
                silver_answers_list = process_answer_string(silver_answers_raw)
                if run_idx == 1:
                    all_silver_answers.append(silver_answers_list)

                negative_answers_raw = row.get("negative_answers")
                negative_answers_list = process_answer_string(negative_answers_raw)
                if run_idx == 1:
                    all_negative_answers.append(negative_answers_list)

                is_correct = evaluate_prediction(predicted_letters, gold_answers_list)
                if is_correct:
                    correct_predictions += 1

                model_score, max_score = calculate_custom_score(predicted_letters, gold_answers_list, silver_answers_list)
                total_model_score += model_score
                total_max_score += max_score

                evaluate_and_log_3x2_matrix(
                    gold_answers_list,
                    silver_answers_list,
                    negative_answers_list,
                    predicted_letters,
                    result_row_dict
                )

                for key in run_confusion_totals:
                    run_confusion_totals[key] += result_row_dict[key]

                result_row_dict["gold_answers"] = format_answer_list_for_csv(row.get("gold_answers"))
                result_row_dict["silver_answers"] = format_answer_list_for_csv(row.get("silver_answers"))
                result_row_dict["negative_answers"] = format_answer_list_for_csv(row.get("negative_answers"))
                result_row_dict["predicted_answers"] = ",".join(predicted_letters)
                result_row_dict["is_correct"] = is_correct
                result_row_dict["model_score"] = model_score
                result_row_dict["max_score"] = max_score

            except Exception as e:
                error_msg = f"Error processing row {index} ({par4pc_identifier}): {type(e).__name__}: {e}"
                print(f"\n  [!] Error processing row {index} ({par4pc_identifier}): {error_msg}")
                print(traceback.format_exc())
                errors += 1
                with open(error_log_file, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"Run {run_idx}: {datetime.now().isoformat()} - {error_msg}\n")
                    f_err.write(traceback.format_exc() + "\n\n")
                result_row_dict["error"] = error_msg
                try:
                    gold_answers_list_on_error = process_answer_string(row.get("gold_answers"))
                    silver_answers_list_on_error = process_answer_string(row.get("silver_answers"))
                    _, max_score_on_error = calculate_custom_score([], gold_answers_list_on_error, silver_answers_list_on_error)
                    result_row_dict["max_score"] = max_score_on_error
                    total_max_score += max_score_on_error
                except Exception as score_err:
                    print(f"Error calculating max score on outer error: {score_err}")
                    result_row_dict["max_score"] = 0

            try:
                with open(run_result_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    writer.writerow(result_row_dict)
            except IOError as e:
                print(f"\n  [!] Error writing row {index} to CSV {run_result_file}: {e}")

        exact_match_rate = (correct_predictions / total_par4pcs) * 100 if total_par4pcs > 0 else 0
        all_exact_match_rates.append(exact_match_rate)
        
        custom_score_percentage = 0
        if total_max_score > 0:
            custom_score_percentage = (total_model_score / total_max_score) * 100
        all_custom_scores.append(custom_score_percentage)
        
        all_confusion_matrices.append(run_confusion_totals)

        print(f"\n--- Run {run_idx}/{runs} Results ---")
        print(f"Exact match rate: {exact_match_rate:.2f}%")
        print(f"Custom score: {custom_score_percentage:.2f}% [max 100]")
        print(f"Errors: {errors}")

    if runs > 1:
        min_exact_match = min(all_exact_match_rates)
        max_exact_match = max(all_exact_match_rates)
        avg_exact_match = np.mean(all_exact_match_rates)
        std_exact_match = np.std(all_exact_match_rates)
        
        min_custom_score = min(all_custom_scores)
        max_custom_score = max(all_custom_scores)
        avg_custom_score = np.mean(all_custom_scores)
        std_custom_score = np.std(all_custom_scores)
        
        print("\n" + "="*60)
        print(f"BASELINE RESULTS SUMMARY ({runs} RUNS)")
        print("="*60)
        print(f"Exact Match Rate:")
        print(f"  Minimum: {min_exact_match:.2f}%")
        print(f"  Maximum: {max_exact_match:.2f}%")
        print(f"  Mean:    {avg_exact_match:.2f}%")
        print(f"  Std Dev: {std_exact_match:.2f}%")
        
        print(f"\nCustom Score (max 100):")
        print(f"  Minimum: {min_custom_score:.2f}%")
        print(f"  Maximum: {max_custom_score:.2f}%")
        print(f"  Mean:    {avg_custom_score:.2f}%")
        print(f"  Std Dev: {std_custom_score:.2f}%")
        
        avg_confusion = {key: np.mean([matrix[key] for matrix in all_confusion_matrices]) for key in all_confusion_matrices[0]}
        
        print(f"\nAverage Confusion Matrix (3x2):")
        print(f"               | Predicted Positive | Predicted Negative |")
        print(f"---------------|--------------------|--------------------|")
        print(f"Actual Gold    | {avg_confusion['gold_answer']:<18.2f} | {avg_confusion['gold_negative']:<18.2f} |")
        print(f"Actual Silver  | {avg_confusion['silver_answer']:<18.2f} | {avg_confusion['silver_negative']:<18.2f} |")
        print(f"Actual Negative| {avg_confusion['negative_answer']:<18.2f} | {avg_confusion['negative_negative']:<18.2f} |")
        
        try:
            with open(stats_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["run", "exact_match_rate", "custom_score"])
                for i in range(runs):
                    writer.writerow([i+1, f"{all_exact_match_rates[i]:.2f}", f"{all_custom_scores[i]:.2f}"])
                writer.writerow([])
                writer.writerow(["statistic", "exact_match_rate", "custom_score"])
                writer.writerow(["min", f"{min_exact_match:.2f}", f"{min_custom_score:.2f}"])
                writer.writerow(["max", f"{max_exact_match:.2f}", f"{max_custom_score:.2f}"])
                writer.writerow(["mean", f"{avg_exact_match:.2f}", f"{avg_custom_score:.2f}"])
                writer.writerow(["std", f"{std_exact_match:.2f}", f"{std_custom_score:.2f}"])
                
                writer.writerow([])
                writer.writerow(["average_confusion_matrix"])
                writer.writerow(["category", "predicted_positive", "predicted_negative"])
                writer.writerow(["gold", f"{avg_confusion['gold_answer']:.2f}", f"{avg_confusion['gold_negative']:.2f}"])
                writer.writerow(["silver", f"{avg_confusion['silver_answer']:.2f}", f"{avg_confusion['silver_negative']:.2f}"])
                writer.writerow(["negative", f"{avg_confusion['negative_answer']:.2f}", f"{avg_confusion['negative_negative']:.2f}"])
                
            print(f"\nStatistics saved to: {stats_file}")
        except IOError as e:
            print(f"\nError writing statistics to CSV: {e}")
    
    print(f"\nDetailed results saved in directory: {result_dir}")
    if runs > 1:
        print(f"Individual run results stored in: {runs_dir}")


def run_baseline_2_evaluation(runs: int = 1):
    """
    Baseline 2: Randomly select exactly 1 option from A-H
    runs: number of repetitions to compute statistics
    """
    print(f"Starting Baseline 2 evaluation with Hugging Face dataset {DATASET_NAME} (test split)")
    print(f"Running {runs} times to compute statistics")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        total_par4pcs = len(df)
        print(f"Loaded {total_par4pcs} par4pcs from Hugging Face dataset {DATASET_NAME} (test split).")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return
    
    dataset_name_for_path = DATASET_NAME.replace("/", "_")

    if df.empty:
        print(f"Error: No data found in the dataset.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir_name = f'result_{timestamp}_baseline2_runs{runs}_{dataset_name_for_path}'
    result_dir = Path(__file__).parent / 'result' / result_dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = result_dir / "individual_runs"
    runs_dir.mkdir(exist_ok=True)
    
    stats_file = result_dir / "baseline2_stats.csv"
    error_log_file = result_dir / "error_log.txt"

    all_exact_match_rates = []
    all_custom_scores = []
    all_gold_answers = []
    all_silver_answers = []
    all_negative_answers = []
    all_confusion_matrices = []

    all_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    for run_idx in range(1, runs + 1):
        print(f"\n--- Starting Baseline 2 Run {run_idx}/{runs} ---")
        
        run_result_file = runs_dir / f"run_{run_idx}_results.csv"
        
        csv_header = ["identifier", "claim_number", "application_number",
                    "gold_answers", "silver_answers", "negative_answers",
                    "predicted_answers", "baseline_selection_info", "is_correct", "error",
                    "model_score", "max_score", 
                    "gold_answer", "silver_answer", "negative_answer",
                    "gold_negative", "silver_negative", "negative_negative"]

        try:
            with open(run_result_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
            print(f"Run {run_idx} results CSV file initialized: {run_result_file}")
        except IOError as e:
            print(f"Error initializing CSV file {run_result_file}: {e}")
            continue

        correct_predictions = 0
        errors = 0
        total_model_score = 0
        total_max_score = 0

        run_confusion_totals = {
            "gold_answer": 0, "silver_answer": 0, "negative_answer": 0,
            "gold_negative": 0, "silver_negative": 0, "negative_negative": 0
        }

        for index, row in df.iterrows():
            print(f"\rProcessing Run {run_idx}/{runs}: {index + 1}/{total_par4pcs} ({((index + 1)/total_par4pcs)*100:.1f}%)", end="")

            par4pc_identifier = f"app_{row.get('application_number', 'N/A')}_claim_{row.get('claim_number', 'N/A')}"

            predicted_letter = random.choice(all_options)
            predicted_letters = [predicted_letter]
            baseline_selection_info = f"Randomly selected exactly 1 option: {predicted_letter}"
            
            is_correct = False
            error_message = ""
            model_score = 0
            max_score = 0

            result_row_dict = {
                "identifier": par4pc_identifier,
                "claim_number": row.get("claim_number", "N/A"),
                "application_number": row.get("application_number", "N/A"),
                "gold_answers": "", "silver_answers": "", "negative_answers": "",
                "predicted_answers": "", "baseline_selection_info": baseline_selection_info,
                "is_correct": False, "error": "",
                "model_score": 0, "max_score": 0,
                "gold_answer": 0, "silver_answer": 0, "negative_answer": 0,
                "gold_negative": 0, "silver_negative": 0, "negative_negative": 0
            }

            try:
                gold_answers_raw = row.get("gold_answers")
                gold_answers_list = process_answer_string(gold_answers_raw)
                
                if run_idx == 1:
                    all_gold_answers.append(gold_answers_list)

                silver_answers_raw = row.get("silver_answers")
                silver_answers_list = process_answer_string(silver_answers_raw)
                if run_idx == 1:
                    all_silver_answers.append(silver_answers_list)

                negative_answers_raw = row.get("negative_answers")
                negative_answers_list = process_answer_string(negative_answers_raw)
                if run_idx == 1:
                    all_negative_answers.append(negative_answers_list)

                is_correct = evaluate_prediction(predicted_letters, gold_answers_list)
                if is_correct:
                    correct_predictions += 1

                model_score, max_score = calculate_custom_score(predicted_letters, gold_answers_list, silver_answers_list)
                total_model_score += model_score
                total_max_score += max_score

                evaluate_and_log_3x2_matrix(
                    gold_answers_list,
                    silver_answers_list,
                    negative_answers_list,
                    predicted_letters,
                    result_row_dict
                )

                for key in run_confusion_totals:
                    run_confusion_totals[key] += result_row_dict[key]

                result_row_dict["gold_answers"] = format_answer_list_for_csv(row.get("gold_answers"))
                result_row_dict["silver_answers"] = format_answer_list_for_csv(row.get("silver_answers"))
                result_row_dict["negative_answers"] = format_answer_list_for_csv(row.get("negative_answers"))
                result_row_dict["predicted_answers"] = ",".join(predicted_letters)
                result_row_dict["is_correct"] = is_correct
                result_row_dict["model_score"] = model_score
                result_row_dict["max_score"] = max_score

            except Exception as e:
                error_msg = f"Error processing row {index} ({par4pc_identifier}): {type(e).__name__}: {e}"
                print(f"\n  [!] Error processing row {index} ({par4pc_identifier}): {error_msg}")
                print(traceback.format_exc())
                errors += 1
                with open(error_log_file, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"Run {run_idx}: {datetime.now().isoformat()} - {error_msg}\n")
                    f_err.write(traceback.format_exc() + "\n\n")
                result_row_dict["error"] = error_msg
                try:
                    gold_answers_list_on_error = process_answer_string(row.get("gold_answers"))
                    silver_answers_list_on_error = process_answer_string(row.get("silver_answers"))
                    _, max_score_on_error = calculate_custom_score([], gold_answers_list_on_error, silver_answers_list_on_error)
                    result_row_dict["max_score"] = max_score_on_error
                    total_max_score += max_score_on_error
                except Exception as score_err:
                    print(f"Error calculating max score on outer error: {score_err}")
                    result_row_dict["max_score"] = 0

            try:
                with open(run_result_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    writer.writerow(result_row_dict)
            except IOError as e:
                print(f"\n  [!] Error writing row {index} to CSV {run_result_file}: {e}")

        exact_match_rate = (correct_predictions / total_par4pcs) * 100 if total_par4pcs > 0 else 0
        all_exact_match_rates.append(exact_match_rate)
        
        custom_score_percentage = 0
        if total_max_score > 0:
            custom_score_percentage = (total_model_score / total_max_score) * 100
        all_custom_scores.append(custom_score_percentage)
        
        all_confusion_matrices.append(run_confusion_totals)

        print(f"\n--- Run {run_idx}/{runs} Results ---")
        print(f"Exact match rate: {exact_match_rate:.2f}%")
        print(f"Custom score: {custom_score_percentage:.2f}% [max 100]")
        print(f"Errors: {errors}")

    if runs > 1:
        min_exact_match = min(all_exact_match_rates)
        max_exact_match = max(all_exact_match_rates)
        avg_exact_match = np.mean(all_exact_match_rates)
        std_exact_match = np.std(all_exact_match_rates)
        
        min_custom_score = min(all_custom_scores)
        max_custom_score = max(all_custom_scores)
        avg_custom_score = np.mean(all_custom_scores)
        std_custom_score = np.std(all_custom_scores)
        
        print("\n" + "="*60)
        print(f"BASELINE 2 RESULTS SUMMARY ({runs} RUNS)")
        print("="*60)
        print(f"Exact Match Rate:")
        print(f"  Minimum: {min_exact_match:.2f}%")
        print(f"  Maximum: {max_exact_match:.2f}%")
        print(f"  Mean:    {avg_exact_match:.2f}%")
        print(f"  Std Dev: {std_exact_match:.2f}%")
        
        print(f"\nCustom Score (max 100):")
        print(f"  Minimum: {min_custom_score:.2f}%")
        print(f"  Maximum: {max_custom_score:.2f}%")
        print(f"  Mean:    {avg_custom_score:.2f}%")
        print(f"  Std Dev: {std_custom_score:.2f}%")
        
        avg_confusion = {key: np.mean([matrix[key] for matrix in all_confusion_matrices]) for key in all_confusion_matrices[0]}
        
        print(f"\nAverage Confusion Matrix (3x2):")
        print(f"               | Predicted Positive | Predicted Negative |")
        print(f"---------------|--------------------|--------------------|")
        print(f"Actual Gold    | {avg_confusion['gold_answer']:<18.2f} | {avg_confusion['gold_negative']:<18.2f} |")
        print(f"Actual Silver  | {avg_confusion['silver_answer']:<18.2f} | {avg_confusion['silver_negative']:<18.2f} |")
        print(f"Actual Negative| {avg_confusion['negative_answer']:<18.2f} | {avg_confusion['negative_negative']:<18.2f} |")
        
        try:
            with open(stats_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["run", "exact_match_rate", "custom_score"])
                for i in range(runs):
                    writer.writerow([i+1, f"{all_exact_match_rates[i]:.2f}", f"{all_custom_scores[i]:.2f}"])
                writer.writerow([])
                writer.writerow(["statistic", "exact_match_rate", "custom_score"])
                writer.writerow(["min", f"{min_exact_match:.2f}", f"{min_custom_score:.2f}"])
                writer.writerow(["max", f"{max_exact_match:.2f}", f"{max_custom_score:.2f}"])
                writer.writerow(["mean", f"{avg_exact_match:.2f}", f"{avg_custom_score:.2f}"])
                writer.writerow(["std", f"{std_exact_match:.2f}", f"{std_custom_score:.2f}"])
                
                writer.writerow([])
                writer.writerow(["average_confusion_matrix"])
                writer.writerow(["category", "predicted_positive", "predicted_negative"])
                writer.writerow(["gold", f"{avg_confusion['gold_answer']:.2f}", f"{avg_confusion['gold_negative']:.2f}"])
                writer.writerow(["silver", f"{avg_confusion['silver_answer']:.2f}", f"{avg_confusion['silver_negative']:.2f}"])
                writer.writerow(["negative", f"{avg_confusion['negative_answer']:.2f}", f"{avg_confusion['negative_negative']:.2f}"])
                
            print(f"\nStatistics saved to: {stats_file}")
        except IOError as e:
            print(f"\nError writing statistics to CSV: {e}")
    
    print(f"\nDetailed results saved in directory: {result_dir}")
    if runs > 1:
        print(f"Individual run results stored in: {runs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cited patent prediction benchmark test using DxD-Lab/PANORAMA-PAR4PC-Bench dataset from Hugging Face.')
    parser.add_argument('--provider', type=str, required=False,
                       choices=['openai', 'anthropic', 'google'],
                       help='LLM provider (openai, anthropic, or google). Required if not using baseline mode.')
    parser.add_argument('--model', type=str, required=False,
                       help='Model name (e.g., gpt-4o, claude-3-opus-20240229, gemini-2.0-flash). Required if not using baseline mode.')
    parser.add_argument('--prompt_mode', type=str, default='zero-shot',
                    choices=['zero-shot', 'cot', 'cot_base'],
                    help="Prompt style: 'zero-shot' (default) or 'cot'")
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline evaluation (random selection of 1+ options from A-H)')
    parser.add_argument('--baseline-2', action='store_true',
                        help='Run baseline 2 evaluation (random selection of exactly 1 option from A-H)')
    parser.add_argument('--runs', type=int, default=20,
                        help='Number of baseline runs to perform (default: 20)')

    args = parser.parse_args()

    try:
        if args.baseline:
            run_baseline_evaluation(args.runs)
        elif args.baseline_2:
            run_baseline_2_evaluation(args.runs)
        else:
            if not args.provider or not args.model:
                parser.error("--provider and --model are required when not using baseline mode.")
            main(args.provider, args.model, args.prompt_mode)
            
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        print(traceback.format_exc())
