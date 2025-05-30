# example(zero-shot): python benchmarks/pi4pc/inference.py --provider google --model gemini-1.5-flash-latest --prompt_mode zero-shot
# example(cot): python benchmarks/pi4pc/inference.py --provider openai --model gpt-4o --prompt_mode cot

# example(baseline): python benchmarks/pi4pc/inference.py --baseline-1
# example(baseline-2): python benchmarks/pi4pc/inference.py --baseline-2

import json
import warnings
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import traceback
import time
import re
import numpy as np
import csv
import random
import sys # For exit
from pydantic import BaseModel, Field
from datasets import load_dataset

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

warnings.filterwarnings(action='ignore')

DATASET_NAME = "DxD-Lab/PANORAMA-PI4PC-Bench"

class ZeroShotParagraphAnswer(BaseModel):
    answer: int = Field(description="The single integer key of the most relevant paragraph.")

class CoTParagraphAnswer(BaseModel):
    reason: str = Field(description="Step-by-step reasoning for selecting the paragraph.")
    answer: int = Field(description="The single integer key of the most relevant paragraph based on the reasoning.")

def get_chat_model(provider: str, model_name: str, prompt_mode: str):
    """채팅 모델 인스턴스를 생성하여 반환합니다."""
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
                    "answer": {"type": "integer"},
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
            structured_llm = chat_model_base.with_structured_output(CoTParagraphAnswer)
        else: # zero-shot
            structured_llm = chat_model_base.with_structured_output(ZeroShotParagraphAnswer)

        return structured_llm 

    else:
        raise ValueError(f"Unsupported provider: {provider}")

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "testing_prompt_citedParagraph.txt"

def create_paragraph_prompt(benchmark_data: Dict[str, Any], prompt_mode: str) -> str:
    if not PROMPT_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Prompt template file not found at: {PROMPT_TEMPLATE_PATH}")

    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    context_raw = benchmark_data.get("context", {})
    prior_art_raw = benchmark_data.get("prior_art_specification", {})
    options_raw = benchmark_data.get("options", {})
    try:
        context = json.loads(context_raw) if isinstance(context_raw, str) else context_raw
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
         if target_claim_text == 'Claim text not found' and str(claim_num).isdigit():
             claim_index = int(claim_num) - 1
             if 0 <= claim_index < len(context['claims']) and isinstance(context['claims'][claim_index], str):
                 target_claim_text = context['claims'][claim_index]


    options_text_list = []
    try:
        sorted_options = sorted(options.items(), key=lambda item: int(str(item[0])))
    except (ValueError, TypeError):
        print(f"Warning: Could not sort options by integer key for app {app_num}, claim {claim_num}. Using default order.")
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
        if prompt_mode == "cot":
            filled_prompt += f"""
**Step-by-Step Method**
*Use the Broadest-Reasonable-Interpretation (BRI) standard throughout.*

1.  **BRI Claim Deconstruction**
    • Break the claim {claim_num} into **numbered limitations** (e.g., [1A]-[1F]).
    • Write each limitation in examiner-style "element : function / relationship" form.
    • Try to include as much of the claim as possible.

2.  **Key Distinguishing Feature(s)**
    • Identify which limitation(s) the applicant asserts as novel / non-obvious.
    • List all the features that should be considered when evaluating novelty and non-obviousness.

3.  **Prior-Art Mapping Table (one table per option paragraph)**
    • For each of the five option paragraphs, provide a detailed mapping to Claim {claim_num} elements.
    • Use the table format to score the degree of overlap between each option paragraph and the claim limitations.
    • IMPORTANT: **Do not skip any options** — evaluate all five paragraphs.
    | Opt# | Maps to elements | Exact term / BRI synonym | Col-Line (or ¶) | Match score* |
    |------|-----------------|--------------------------|-----------------|-------------|
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    *Scoring: 0 = missing, 1 = mention, 2 = partial, 3 = full & explicit.*

4.  **Select the Most Relevant Paragraph for Patentability Evaluation**
    • Your goal is to identify exactly ONE paragraph most relevant to evaluate the novelty/non-obviousness of the applicant's claimed invention.
    • Select exactly one paragraph based on its relevance to the novelty or non-obviousness of the Key Distinguishing Features (KD-x) in Claim {claim_num}.
    • Do not select multiple keys or provide general reasoning.
    • Focus on technical relevance, improvements, and system integration when selecting your paragraph.
    Selection Criteria:
    • Consider paragraphs that scored ≥ 1 points in Step 3.
    • Technical Objectives: Does the paragraph directly support the technical objectives of the Claim? Does it provide a solution to the problem presented by the Claim?
    • Prior Art Improvements: Does the paragraph present innovative improvements to existing systems or technologies?
    • System Integration: Does the paragraph explain how elements of the system described in the Claim interact or integrate with each other?
    • Motivation to Combine: Does the paragraph offer a motivational context for combining features, particularly for a §103 rejection?

    Output Requirements:
    • Clearly indicate your final selection as the Primary Reference (PR).
    • Provide a concise reason for your selection based strictly on the criteria above.

5. **Consistency & Inherency Check**
   • Verify the selected paragraph does not contradict any claim limitation.

6. **Output (JSON only)**
    Always write the "reason" first and then write the "answer".
   • 'reason' MUST list Step1-Step6 in order, each separated by ';'.
     ▸ Step1 <statutory/context> ;
     ▸ Step2 <limits> ;
     ▸ Step3 <key feature> ;
     ▸ Step4 <mapping & score> ;
     ▸ Step5 <rank/tie-break> ;
     ▸ Step6 <consistency/inherency & §102 or §103 result>.
   • Keep "reason" ≤ 1000 words.
   • "answer" = single paragraph key (int).

```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ... ; Step4 ...; Step5 ...; Step6 ...","answer": 17}}
```
"""
        if prompt_mode == "cot_base":
            filled_prompt += f"""
Think through the steps required to evaluate this, craft the supporting rationale accordingly, and then deliver your answer based on that rationale.
Always write the "reason" **first** and then write the "answer".

Answer format (JSON only):
```json
{{"reason":"...","answer": 17}}
```
"""  

        else: # --- zero‑shot ---
            filled_prompt += """
Answer format (JSON only)
Return ONLY this JSON object - DO NOT INCLUDE ANY REASON IN THE ANSWER.
```json
{{"answer": ##}}
```
"""
        return filled_prompt
    except KeyError as e:
        raise ValueError(f"Missing key in prompt template or format data: {e}")


def process_llm_response_paragraphs(response_data: Union[dict, BaseModel, None]) -> Optional[int]:
    """Extracts a single integer key from the LLM's response (Pydantic model or dict)."""
    answer = None
    if response_data is None:
        return None

    try:
        if isinstance(response_data, BaseModel):
            answer = response_data.answer
        elif isinstance(response_data, dict):
            answer = response_data.get("answer")
        else:
            print(f"Warning: Unexpected response type for process_llm_response_paragraphs: {type(response_data)}")
            return None

        if answer is not None:
            return int(answer)
        else:
            print(f"Warning: 'answer' key not found or is None in response: {response_data}")
            return None

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not extract integer 'answer' from response: {response_data}. Error: {e}")
        return None


def calculate_paragraph_score(
    predicted_key: Optional[int],
    gold_keys: List[int],
    silver_keys: List[int],
    all_option_keys: List[int]
) -> Dict[str, Any]:
    """Calculates the score based on the predicted key and Gold/Silver/Negative sets."""
    score = 0
    category = "Invalid Response (Non-Integer)"

    if predicted_key is None:
        return {"score": 0, "category": category, "is_valid_option_key": False}

    is_valid_option_key = predicted_key in all_option_keys

    if predicted_key in gold_keys:
        score = 2
        category = "Gold"
    elif predicted_key in silver_keys:
        score = 1
        category = "Silver"
    elif is_valid_option_key:
        score = 0
        category = "Negative"
    else:
        score = 0
        category = "Invalid Key (Not in Options)"

    return {"score": score, "category": category, "is_valid_option_key": is_valid_option_key}


def format_int_list_for_csv(answer_data):
    if not answer_data: return ""
    try:
        int_list = sorted([int(k) for k in answer_data])
        return ','.join(map(str, int_list))
    except (ValueError, TypeError):
        return str(answer_data)

def calculate_3x2_matrix(
    gold_keys: List[int],
    silver_keys: List[int],
    negative_keys: List[int],
    predicted_keys: List[int]
) -> dict:

    gold_set     = set(gold_keys)
    silver_set   = set(silver_keys)
    negative_set = set(negative_keys)
    pred_set     = set(predicted_keys)

    return {
        "gold_answer"      : len(pred_set & gold_set),
        "silver_answer"    : len(pred_set & silver_set),
        "negative_answer"  : len(pred_set & negative_set),

        "gold_negative"    : len(gold_set     - pred_set),
        "silver_negative"  : len(silver_set   - pred_set),
        "negative_negative": len(negative_set - pred_set),
    }


def evaluate_and_log_3x2_matrix(gold_keys: List[int], silver_keys: List[int], negative_keys: List[int], predicted_keys: List[int], result_row_dict: dict):
    matrix = calculate_3x2_matrix(gold_keys, silver_keys, negative_keys, predicted_keys)

    result_row_dict["gold_answer"] = matrix["gold_answer"]
    result_row_dict["silver_answer"] = matrix["silver_answer"]
    result_row_dict["negative_answer"] = matrix["negative_answer"]
    result_row_dict["gold_negative"] = matrix["gold_negative"]
    result_row_dict["silver_negative"] = matrix["silver_negative"]
    result_row_dict["negative_negative"] = matrix["negative_negative"]


def run_llm_evaluation(args: argparse.Namespace):
    """LLM 기반 Cited Paragraph 벤치마크를 실행하고 결과를 저장합니다."""
    provider = args.provider
    model_name = args.model
    prompt_mode = args.prompt_mode

    print(f"Starting LLM-based Cited Paragraph benchmark test with {provider} - {model_name} (Prompt: {prompt_mode})")
    print(f"Loading benchmark data from Hugging Face dataset {DATASET_NAME} (test split)\n")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
    except Exception as e:
        print(f"❌ Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return

    total_items = len(benchmark_data_list)
    if total_items == 0:
        print("❌ No rows found in benchmark file"); return
    print(f"Loaded {total_items} items.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    results_dir = result_dir / f"result_{timestamp}_{provider}_{model_name}_{prompt_mode}_{dataset_name_for_path}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "evaluation_results.csv"
    error_log = results_dir / "error_log.txt"

    fieldnames = [
        "identifier","application_number","claim_number","cited_patent_id",
        "predicted_key","gold_keys","silver_keys","negative_keys",
        "score","category","is_valid_option_key","llm_raw_response","error",
        "gold_answer","silver_answer","negative_answer",
        "gold_negative","silver_negative","negative_negative"
    ]
    with open(results_csv,"w",newline="",encoding="utf-8-sig") as f:
        csv.DictWriter(f,fieldnames).writeheader()

    chat = get_chat_model(provider, model_name, prompt_mode)
    confusion_totals = {
        "gold_answer":0, "silver_answer":0, "negative_answer":0,
        "gold_negative":0, "silver_negative":0, "negative_negative":0
    }
    stats = {
        "processed":0, "errors":0,
        "gold":0,"silver":0,"negative":0,
        "invalid_key":0,"invalid_resp":0,
        "score_sum":0
    }

    for idx,row in enumerate(benchmark_data_list):
        stats["processed"] += 1
        app_num   = row.get("application_number","N/A")
        claim_num = row.get("claim_number","N/A")

        prior_raw = row.get("prior_art_specification",{})
        try:
            prior = json.loads(prior_raw) if isinstance(prior_raw,str) else prior_raw
            cited_patent_id = prior.get("patent_id","N/A") if isinstance(prior,dict) else "N/A"
        except:
            cited_patent_id = "N/A"
        identifier = f"app{app_num}_cl{claim_num}_pa{cited_patent_id}_{idx}"

        progress_percent = ((idx + 1) / total_items) * 100
        print(f"Processing LLM: {idx + 1}/{total_items} ({progress_percent:.1f}%) - {identifier}...", end='\r')

        gold_keys    = [int(k) for k in row.get("gold_answers",[])]
        silver_keys  = [int(k) for k in row.get("silver_answers",[])]
        negative_keys= [int(k) for k in row.get("negative_answers",[])]

        opts_raw = row.get("options",{})
        option_keys_for_eval = []
        try:
            opts = json.loads(opts_raw) if isinstance(opts_raw,str) else opts_raw
            if isinstance(opts, dict):
                sorted_option_items = sorted(opts.items(), key=lambda item: int(str(item[0])))
                option_keys_for_eval = [int(item[0]) for item in sorted_option_items]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"\nWarning: Could not parse options for {identifier}: {e}. Options: {opts_raw}")
            option_keys_for_eval = []


        prompt = create_paragraph_prompt(row, prompt_mode)
        llm_response_object = None
        llm_raw_response_str = ""
        predicted_key=None; error=""; success=False

        for attempt in range(3): # LLM call retry loop
            try:
                system_message = SystemMessage(content="You are a patent expert identifying the single most relevant cited paragraph key. Respond in the requested JSON format.")
                human_message = HumanMessage(content=prompt)

                if provider.lower() == 'google':
                    llm_response_object = chat.invoke([system_message, human_message])
                    llm_raw_response_str = llm_response_object.model_dump_json()
                elif provider.lower() == 'anthropic':
                    resp = chat([system_message, human_message])
                    if resp.tool_calls:
                         llm_response_object = resp.tool_calls[0]["args"]
                         llm_raw_response_str = json.dumps(llm_response_object)
                    else:
                         llm_response_object = {"answer": None}
                         llm_raw_response_str = resp.content.strip()
                         print(f"\nWarning: Anthropic response did not contain tool calls for {identifier}: {llm_raw_response_str}")
                else:
                    resp = chat([system_message, human_message])
                    llm_raw_response_str = resp.content.strip()
                    try:
                        llm_response_object = json.loads(llm_raw_response_str)
                    except json.JSONDecodeError:
                        print(f"\nWarning: Could not parse OpenAI response as JSON for {identifier}: {llm_raw_response_str}")
                        llm_response_object = {"answer": None}

                predicted_key = process_llm_response_paragraphs(llm_response_object)
                success=True; break

            except Exception as e:
                error = f"LLM fail attempt {attempt+1} for {identifier}: {type(e).__name__}: {str(e)[:200]}"
                llm_raw_response_str = f"ERROR: {error}"
                try:
                    with open(error_log, 'a', encoding='utf-8') as f_err:
                        f_err.write(f"--- Error @ {datetime.now().isoformat()} for identifier: {identifier}, Attempt: {attempt+1} ---\n")
                        f_err.write(f"LLM Response Object before error: {type(llm_response_object)}: {str(llm_response_object)[:500]}\n")
                        f_err.write(f"Exception Type: {type(e).__name__}\n")
                        f_err.write(f"Exception Msg: {str(e)[:500]}\n")
                        f_err.write("Traceback:\n")
                        traceback.print_exc(file=f_err)
                        f_err.write("-" * 50 + "\n\n")
                except IOError as log_err:
                    print(f"\n[!] Critical: Failed to write to error log file {error_log}: {log_err}")

                if attempt<2:
                    wait_time = 2**attempt
                    print(f"\n  [!] {error}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n  [!] Max retries reached for {identifier}. Last error: {error}")

        if not success:
            stats["errors"] += 1
            predicted_key = None 

        eval_res = calculate_paragraph_score(predicted_key, gold_keys, silver_keys, option_keys_for_eval)
        
        stats["score_sum"] += eval_res["score"]
        cat = eval_res["category"]
        if   cat=="Gold":   stats["gold"]   +=1
        elif cat=="Silver": stats["silver"] +=1
        elif cat=="Negative": stats["negative"] +=1
        elif cat=="Invalid Key (Not in Options)": stats["invalid_key"] +=1
        elif cat=="Invalid Response (Non-Integer)": stats["invalid_resp"] +=1

        cm = calculate_3x2_matrix(gold_keys, silver_keys, negative_keys, [predicted_key] if predicted_key is not None else [])
        for k in confusion_totals: confusion_totals[k] += cm[k]

        row_dict = {
            "identifier":identifier,
            "application_number":app_num,
            "claim_number":claim_num,
            "cited_patent_id":cited_patent_id,
            "predicted_key":predicted_key if predicted_key is not None else "",
            "gold_keys":format_int_list_for_csv(gold_keys),
            "silver_keys":format_int_list_for_csv(silver_keys),
            "negative_keys":format_int_list_for_csv(negative_keys),
            "score":eval_res["score"],
            "category":eval_res["category"],
            "is_valid_option_key":eval_res["is_valid_option_key"],
            "llm_raw_response":llm_raw_response_str,
            "error":error if not success else "",
            "gold_answer":cm["gold_answer"],
            "silver_answer":cm["silver_answer"],
            "negative_answer":cm["negative_answer"],
            "gold_negative":cm["gold_negative"],
            "silver_negative":cm["silver_negative"],
            "negative_negative":cm["negative_negative"]
        }
        try:
            with open(results_csv,"a",newline="",encoding="utf-8-sig") as f:
                csv.DictWriter(f,fieldnames).writerow(row_dict)
        except IOError as csv_err:
             print(f"\n[!] Critical: Failed to write row {idx+1} to CSV {results_csv}: {csv_err}")

    print()

    total_row = {k:"" for k in fieldnames}
    total_row["identifier"]="TOTAL"
    total_row.update(confusion_totals)
    with open(results_csv,"a",newline="",encoding="utf-8-sig") as f:
        csv.DictWriter(f,fieldnames).writerow(total_row)

    valid_attempts = stats["processed"] - stats["errors"]
    max_possible_score = valid_attempts * 2 if valid_attempts > 0 else 1

    print("\n--- LLM Evaluation Summary -------------------------------------")
    print(f"Processed: {stats['processed']}    LLM Errors (after retries): {stats['errors']}")
    avg_score = stats['score_sum'] / max(valid_attempts, 1)
    percent_score = (stats['score_sum'] / max(max_possible_score, 1)) * 100
    print(f"Average score (on {valid_attempts} valid items): {avg_score:.2f} / 2  ({percent_score:.2f}% of max possible score)")
    print(f"Gold  🥇: {stats['gold']}  | Silver 🥈: {stats['silver']}  | Neg ❌: {stats['negative']}")
    print(f"Invalid key (not in options): {stats['invalid_key']}  | Invalid resp (non-int/parse error): {stats['invalid_resp']}")
    print("Confusion totals:")
    print(f"               | Predicted Positive | Predicted Negative |")
    print(f"---------------|--------------------|--------------------|")
    print(f"Actual Gold    | {confusion_totals['gold_answer']:<18} | {confusion_totals['gold_negative']:<18} |")
    print(f"Actual Silver  | {confusion_totals['silver_answer']:<18} | {confusion_totals['silver_negative']:<18} |")
    print(f"Actual Negative| {confusion_totals['negative_answer']:<18} | {confusion_totals['negative_negative']:<18} |")
    print("\nDetailed CSV →", results_csv)
    if stats["errors"] > 0:
        print(f"Error details (LLM calls) logged to: {error_log}")

    print(f"\nDetailed results saved in: {results_dir}")
    if os.path.exists(error_log) and os.path.getsize(error_log) > 0:
         print(f"Error details logged to: {error_log}")
    elif stats["errors"] > 0:
         print(f"Processing errors occurred, check terminal output for details.")


def run_baseline_1_evaluation(args: argparse.Namespace):
    """Baseline 1: Random choice"""
    num_runs = args.baseline_1_runs
    print(f"Starting Baseline 1 (Random Choice) evaluation using Hugging Face dataset {DATASET_NAME} (test split)")
    print(f"Will run {num_runs} times to compute statistics.")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
    except Exception as e:
        print(f"❌ Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return

    total_items = len(benchmark_data_list)
    if total_items == 0:
        print("❌ No rows found in benchmark file"); return
    print(f"Loaded {total_items} items.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    results_dir = result_dir / f"result_{timestamp}_baseline1_runs{num_runs}_{dataset_name_for_path}"
    results_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = results_dir / "individual_runs"
    runs_dir.mkdir(exist_ok=True)

    error_log = results_dir / "data_error_log.txt"

    stats_csv = results_dir / "statistics_summary.csv"

    all_runs_stats = []
    all_runs_scores = []
    all_runs_gold_counts = []
    all_runs_silver_counts = []
    all_runs_negative_counts = []

    fieldnames = [
        "identifier","application_number","claim_number","cited_patent_id",
        "predicted_key","gold_keys","silver_keys","negative_keys",
        "score","category","is_valid_option_key","llm_raw_response","error",
        "gold_answer","silver_answer","negative_answer",
        "gold_negative","silver_negative","negative_negative"
    ]

    for run_index in range(num_runs):
        print(f"\n--- Starting Baseline 1 Run {run_index + 1}/{num_runs} ---")
        
        current_run_csv = runs_dir / f"run_{run_index + 1}.csv"
        
        confusion_totals = {"gold_answer":0, "silver_answer":0, "negative_answer":0, "gold_negative":0, "silver_negative":0, "negative_negative":0}
        stats = {"processed":0, "data_errors":0, "gold":0,"silver":0,"negative":0, "invalid_key":0,"invalid_resp":0, "score_sum":0, "no_options":0}

        with open(current_run_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames).writeheader()

            for idx, row in enumerate(benchmark_data_list):
                stats["processed"] += 1
                app_num = row.get("application_number", "N/A")
                claim_num = row.get("claim_number", "N/A")
                prior_raw = row.get("prior_art_specification", {})
                try:
                    prior = json.loads(prior_raw) if isinstance(prior_raw, str) else prior_raw
                    cited_patent_id = prior.get("patent_id", "N/A") if isinstance(prior, dict) else "N/A"
                except: cited_patent_id = "N/A"
                identifier = f"app{app_num}_cl{claim_num}_pa{cited_patent_id}_{idx}"

                progress_percent = ((idx + 1) / total_items) * 100
                print(f"Run {run_index + 1}: Processing {idx + 1}/{total_items} ({progress_percent:.1f}%)...", end='\r')

                gold_keys = [int(k) for k in row.get("gold_answers", [])]
                silver_keys = [int(k) for k in row.get("silver_answers", [])]
                negative_keys = [int(k) for k in row.get("negative_answers", [])]

                opts_raw = row.get("options", {})
                current_item_option_keys = []
                error_msg_data = ""
                try:
                    opts = json.loads(opts_raw) if isinstance(opts_raw, str) else opts_raw
                    if isinstance(opts, dict) and opts:
                        sorted_option_items = sorted(opts.items(), key=lambda item: int(str(item[0])))
                        current_item_option_keys = [int(item[0]) for item in sorted_option_items]
                    elif not opts:
                        error_msg_data = f"No options provided for {identifier}"
                        stats["no_options"] += 1
                    else:
                        error_msg_data = f"Options for {identifier} are not a dict: {type(opts_raw)}"
                        stats["data_errors"] += 1
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    error_msg_data = f"Error parsing options for {identifier}: {e}. Options: {str(opts_raw)[:100]}"
                    stats["data_errors"] += 1
                    with open(error_log, 'a', encoding='utf-8') as f_err:
                        f_err.write(f"Run {run_index + 1}, {identifier}: {error_msg_data}\n")

                predicted_key = None
                llm_raw_response_str = f"Baseline 1 (Run {run_index + 1}): Random choice"

                if current_item_option_keys:
                    predicted_key = random.choice(current_item_option_keys)
                else:
                    if not error_msg_data:
                        llm_raw_response_str = f"Baseline 1 (Run {run_index + 1}): No options available for {identifier}"

                eval_res = calculate_paragraph_score(predicted_key, gold_keys, silver_keys, current_item_option_keys)
                stats["score_sum"] += eval_res["score"]
                cat = eval_res["category"]
                if cat == "Gold": stats["gold"] += 1
                elif cat == "Silver": stats["silver"] += 1
                elif cat == "Negative": stats["negative"] += 1
                elif cat == "Invalid Key (Not in Options)": stats["invalid_key"] += 1
                elif cat == "Invalid Response (Non-Integer)": stats["invalid_resp"] += 1

                cm = calculate_3x2_matrix(gold_keys, silver_keys, negative_keys, [predicted_key] if predicted_key is not None else [])
                for k_cm in confusion_totals: confusion_totals[k_cm] += cm[k_cm]

                final_error_for_row = error_msg_data
                if predicted_key is None and not current_item_option_keys and not final_error_for_row:
                    final_error_for_row = "No options to choose from"

                row_dict = {
                    "identifier": identifier, "application_number": app_num, "claim_number": claim_num, "cited_patent_id": cited_patent_id,
                    "predicted_key": predicted_key if predicted_key is not None else "",
                    "gold_keys": format_int_list_for_csv(gold_keys), "silver_keys": format_int_list_for_csv(silver_keys), "negative_keys": format_int_list_for_csv(negative_keys),
                    "score": eval_res["score"], "category": eval_res["category"], "is_valid_option_key": eval_res["is_valid_option_key"],
                    "llm_raw_response": llm_raw_response_str if not final_error_for_row else f"Baseline 1 (Run {run_index + 1}): {final_error_for_row}",
                    "error": final_error_for_row,
                    "gold_answer": cm["gold_answer"], "silver_answer": cm["silver_answer"], "negative_answer": cm["negative_answer"],
                    "gold_negative": cm["gold_negative"], "silver_negative": cm["silver_negative"], "negative_negative": cm["negative_negative"]
                }
                csv.DictWriter(f, fieldnames).writerow(row_dict)

            total_row = {k: "" for k in fieldnames}
            total_row["identifier"] = f"TOTAL (Run {run_index + 1})"
            total_row.update(confusion_totals)
            csv.DictWriter(f, fieldnames).writerow(total_row)

        valid_for_avg = stats["processed"] - stats["data_errors"]
        max_score = valid_for_avg * 2 if valid_for_avg > 0 else 1
        avg_score_b1 = stats['score_sum'] / max(valid_for_avg, 1)
        percent_score_b1 = (stats['score_sum'] / max(max_score, 1)) * 100

        run_summary = {
            "run_index": run_index + 1,
            "processed": stats["processed"],
            "data_errors": stats["data_errors"],
            "no_options": stats["no_options"],
            "gold": stats["gold"],
            "silver": stats["silver"],
            "negative": stats["negative"],
            "score_sum": stats["score_sum"],
            "avg_score": avg_score_b1,
            "percent_score": percent_score_b1,
            "confusion_totals": confusion_totals
        }
        all_runs_stats.append(run_summary)
        all_runs_scores.append(avg_score_b1)
        all_runs_gold_counts.append(stats["gold"])
        all_runs_silver_counts.append(stats["silver"])
        all_runs_negative_counts.append(stats["negative"])

        print(f"\nRun {run_index + 1} completed: Avg Score = {avg_score_b1:.3f} / 2 ({percent_score_b1:.2f}% of max)")
        print(f"Gold: {stats['gold']}  |  Silver: {stats['silver']}  |  Negative: {stats['negative']}")

    scores_array = np.array(all_runs_scores)
    gold_array = np.array(all_runs_gold_counts)
    silver_array = np.array(all_runs_silver_counts)
    negative_array = np.array(all_runs_negative_counts)

    stats_summary = {
        "min_score": np.min(scores_array),
        "max_score": np.max(scores_array),
        "mean_score": np.mean(scores_array),
        "median_score": np.median(scores_array),
        "std_dev_score": np.std(scores_array),
        
        "min_gold": np.min(gold_array),
        "max_gold": np.max(gold_array),
        "mean_gold": np.mean(gold_array),
        "std_dev_gold": np.std(gold_array),
        
        "min_silver": np.min(silver_array),
        "max_silver": np.max(silver_array),
        "mean_silver": np.mean(silver_array),
        "std_dev_silver": np.std(silver_array),
        
        "min_negative": np.min(negative_array),
        "max_negative": np.max(negative_array),
        "mean_negative": np.mean(negative_array),
        "std_dev_negative": np.std(negative_array),
    }

    with open(stats_csv, "w", newline="", encoding="utf-8-sig") as f_stats:
        stats_writer = csv.writer(f_stats)
        stats_writer.writerow(["Statistic", "Value"])
        stats_writer.writerow(["Number of Runs", num_runs])
        stats_writer.writerow(["Minimum Score", f"{stats_summary['min_score']:.3f}"])
        stats_writer.writerow(["Maximum Score", f"{stats_summary['max_score']:.3f}"])
        stats_writer.writerow(["Mean Score", f"{stats_summary['mean_score']:.3f}"])
        stats_writer.writerow(["Median Score", f"{stats_summary['median_score']:.3f}"])
        stats_writer.writerow(["Score Std Dev", f"{stats_summary['std_dev_score']:.3f}"])
        stats_writer.writerow([])
        stats_writer.writerow(["Min Gold Count", stats_summary['min_gold']])
        stats_writer.writerow(["Max Gold Count", stats_summary['max_gold']])
        stats_writer.writerow(["Mean Gold Count", f"{stats_summary['mean_gold']:.2f}"])
        stats_writer.writerow(["Gold Std Dev", f"{stats_summary['std_dev_gold']:.2f}"])
        stats_writer.writerow([])
        stats_writer.writerow(["Min Silver Count", stats_summary['min_silver']])
        stats_writer.writerow(["Max Silver Count", stats_summary['max_silver']])
        stats_writer.writerow(["Mean Silver Count", f"{stats_summary['mean_silver']:.2f}"])
        stats_writer.writerow(["Silver Std Dev", f"{stats_summary['std_dev_silver']:.2f}"])
        stats_writer.writerow([])
        stats_writer.writerow(["Min Negative Count", stats_summary['min_negative']])
        stats_writer.writerow(["Max Negative Count", stats_summary['max_negative']])
        stats_writer.writerow(["Mean Negative Count", f"{stats_summary['mean_negative']:.2f}"])
        stats_writer.writerow(["Negative Std Dev", f"{stats_summary['std_dev_negative']:.2f}"])

    print("\n" + "="*60)
    print(f"BASELINE 1 RESULTS SUMMARY ({num_runs} RUNS)")
    print("="*60)
    print(f"Score Statistics (out of 2):")
    print(f"  Minimum: {stats_summary['min_score']:.3f}")
    print(f"  Maximum: {stats_summary['max_score']:.3f}")
    print(f"  Mean:    {stats_summary['mean_score']:.3f}")
    print(f"  Median:  {stats_summary['median_score']:.3f}")
    print(f"  Std Dev: {stats_summary['std_dev_score']:.3f}")
    
    print("\nCategory Counts:")
    print(f"  Gold:     {stats_summary['min_gold']} to {stats_summary['max_gold']} (mean: {stats_summary['mean_gold']:.2f} ± {stats_summary['std_dev_gold']:.2f})")
    print(f"  Silver:   {stats_summary['min_silver']} to {stats_summary['max_silver']} (mean: {stats_summary['mean_silver']:.2f} ± {stats_summary['std_dev_silver']:.2f})")
    print(f"  Negative: {stats_summary['min_negative']} to {stats_summary['max_negative']} (mean: {stats_summary['mean_negative']:.2f} ± {stats_summary['std_dev_negative']:.2f})")
    
    print(f"\nDetailed results saved in directory: {result_dir}")
    print(f"Individual run CSVs: {runs_dir}")
    print(f"Statistics summary: {stats_csv}")
    if any(run["data_errors"] > 0 for run in all_runs_stats):
        print(f"Data error details logged to: {error_log}")


def run_baseline_2_evaluation(args: argparse.Namespace):
    """Baseline 2: N-th choice"""
    print(f"Starting Baseline 2 (N-th Choice) evaluation using Hugging Face dataset {DATASET_NAME} (test split)")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
    except Exception as e:
        print(f"❌ Error loading dataset from Hugging Face {DATASET_NAME}: {e}")
        print(traceback.format_exc())
        return
    
    total_items = len(benchmark_data_list)
    if total_items == 0: print("❌ No rows found in benchmark file"); return
    print(f"Loaded {total_items} items.\n")
    
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    
    if df.empty:
        print(f"Error: No data found in the dataset.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_dir_name = f'result_{timestamp}_baseline2_{dataset_name_for_path}'
    results_dir = result_dir / result_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    error_log_file = results_dir / "error_log.txt"

    max_num_options_overall = 0
    parsed_item_data = []

    for idx_setup, row_setup in enumerate(benchmark_data_list):
        temp_identifier = f"item_{idx_setup}"
        temp_opts_raw = row_setup.get("options", {})
        temp_sorted_option_keys = []
        try:
            opts = json.loads(temp_opts_raw) if isinstance(temp_opts_raw, str) else temp_opts_raw
            if isinstance(opts, dict) and opts:
                sorted_items = sorted(opts.items(), key=lambda item: int(str(item[0])))
                temp_sorted_option_keys = [int(item[0]) for item in sorted_items]
                if len(temp_sorted_option_keys) > max_num_options_overall:
                    max_num_options_overall = len(temp_sorted_option_keys)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Could not parse options for item {idx_setup} during B2 setup: {e}")
        
        app_num_setup   = row_setup.get("application_number","N/A")
        claim_num_setup = row_setup.get("claim_number","N/A")
        prior_raw_setup = row_setup.get("prior_art_specification",{})
        cited_patent_id_setup = "N/A"
        try:
            prior_setup = json.loads(prior_raw_setup) if isinstance(prior_raw_setup,str) else prior_raw_setup
            if isinstance(prior_setup,dict): cited_patent_id_setup = prior_setup.get("patent_id","N/A")
        except: pass


        parsed_item_data.append({
            "application_number": app_num_setup,
            "claim_number": claim_num_setup,
            "cited_patent_id_prefix": f"app{app_num_setup}_cl{claim_num_setup}_pa{cited_patent_id_setup}",
            "gold_keys": [int(k) for k in row_setup.get("gold_answers", [])],
            "silver_keys": [int(k) for k in row_setup.get("silver_answers", [])],
            "negative_keys": [int(k) for k in row_setup.get("negative_answers", [])],
            "sorted_option_keys": temp_sorted_option_keys,
            "original_index": idx_setup
        })


    if max_num_options_overall == 0:
        print("❌ No valid options found in any benchmark items for Baseline 2.")
        return

    fieldnames = [
        "identifier","application_number","claim_number","cited_patent_id",
        "predicted_key","gold_keys","silver_keys","negative_keys",
        "score","category","is_valid_option_key","llm_raw_response","error",
        "gold_answer","silver_answer","negative_answer",
        "gold_negative","silver_negative","negative_negative"
    ]

    best_n_for_overall_summary = -1
    highest_avg_score_overall = -1.0
    best_n_detailed_results_for_csv = []
    best_n_confusion_totals_for_csv = {}
    best_n_stats_for_csv = {}

    print(f"Will evaluate for N from 1 to {max_num_options_overall}\n")

    for n_choice_index in range(max_num_options_overall):
        current_n_value = n_choice_index + 1
        print(f"--- Evaluating Baseline 2: Using {current_n_value}-th option for all items ---")

        current_n_run_stats = {"processed":0, "gold":0,"silver":0,"negative":0,
                               "invalid_key":0,"invalid_resp":0, "score_sum":0,
                               "items_without_nth_option":0}
        current_n_run_confusion_totals = {"gold_answer":0, "silver_answer":0, "negative_answer":0,
                                          "gold_negative":0, "silver_negative":0, "negative_negative":0}
        temp_detailed_results_for_this_n_run = []

        for item_idx, item_data in enumerate(parsed_item_data):
            current_n_run_stats["processed"] += 1
            
            identifier = f"{item_data['cited_patent_id_prefix']}_{item_data['original_index']}"
            app_num = item_data["application_number"]
            claim_num = item_data["claim_number"]
            cited_patent_id = item_data["cited_patent_id_prefix"].split("_pa")[-1] if "_pa" in item_data["cited_patent_id_prefix"] else "N/A"


            gold_keys = item_data["gold_keys"]
            silver_keys = item_data["silver_keys"]
            negative_keys = item_data["negative_keys"]
            sorted_option_keys_for_item = item_data["sorted_option_keys"]

            predicted_key = None
            error_msg_b2 = ""
            llm_raw_response_b2 = f"Baseline 2: {current_n_value}-th choice"

            if len(sorted_option_keys_for_item) > n_choice_index:
                predicted_key = sorted_option_keys_for_item[n_choice_index]
            else:
                current_n_run_stats["items_without_nth_option"] += 1
                error_msg_b2 = f"Item only has {len(sorted_option_keys_for_item)} options, cannot pick {current_n_value}-th."


            eval_res = calculate_paragraph_score(predicted_key, gold_keys, silver_keys, sorted_option_keys_for_item)
            current_n_run_stats["score_sum"] += eval_res["score"]
            cat = eval_res["category"]

            if   cat=="Gold":   current_n_run_stats["gold"]   +=1
            elif cat=="Silver": current_n_run_stats["silver"] +=1
            elif cat=="Negative": current_n_run_stats["negative"] +=1
            elif cat=="Invalid Key (Not in Options)": current_n_run_stats["invalid_key"] +=1
            elif cat=="Invalid Response (Non-Integer)": current_n_run_stats["invalid_resp"] +=1

            cm_b2 = calculate_3x2_matrix(gold_keys, silver_keys, negative_keys, [predicted_key] if predicted_key is not None else [])
            for k_cm_b2 in current_n_run_confusion_totals: current_n_run_confusion_totals[k_cm_b2] += cm_b2[k_cm_b2]

            row_data_for_this_item = {
                "identifier":identifier, "application_number":app_num, "claim_number":claim_num, "cited_patent_id":cited_patent_id,
                "predicted_key":predicted_key if predicted_key is not None else "",
                "gold_keys":format_int_list_for_csv(gold_keys),
                "silver_keys":format_int_list_for_csv(silver_keys),
                "negative_keys":format_int_list_for_csv(negative_keys),
                "score":eval_res["score"], "category":eval_res["category"],
                "is_valid_option_key":eval_res["is_valid_option_key"],
                "llm_raw_response": llm_raw_response_b2 if not error_msg_b2 else f"{llm_raw_response_b2} ({error_msg_b2})",
                "error":error_msg_b2,
                "gold_answer":cm_b2["gold_answer"], "silver_answer":cm_b2["silver_answer"], "negative_answer":cm_b2["negative_answer"],
                "gold_negative":cm_b2["gold_negative"], "silver_negative":cm_b2["silver_negative"], "negative_negative":cm_b2["negative_negative"]
            }
            temp_detailed_results_for_this_n_run.append(row_data_for_this_item)
            print(f"  Processed item {item_idx+1}/{total_items} for N={current_n_value}...", end='\r')
        print("\n")

        num_items_for_avg_this_n = total_items
        avg_score_for_this_n = current_n_run_stats["score_sum"] / max(num_items_for_avg_this_n, 1)
        max_possible_score_this_n = num_items_for_avg_this_n * 2 if num_items_for_avg_this_n > 0 else 1
        percent_score_this_n = (current_n_run_stats["score_sum"] / max(max_possible_score_this_n, 1)) * 100

        print(f"Summary for N={current_n_value} choice:")
        print(f"  Total items processed: {total_items}")
        print(f"  Items where {current_n_value}-th option was not available: {current_n_run_stats['items_without_nth_option']}")
        print(f"  Average score: {avg_score_for_this_n:.3f} / 2  ({percent_score_this_n:.2f}% of max)")
        print(f"  Gold: {current_n_run_stats['gold']}  | Silver: {current_n_run_stats['silver']}  | Neg: {current_n_run_stats['negative']}")
        print(f"  Invalid Key (pred not in options): {current_n_run_stats['invalid_key']} | Invalid Resp (pred is None): {current_n_run_stats['invalid_resp']}\n")

        if avg_score_for_this_n > highest_avg_score_overall:
            highest_avg_score_overall = avg_score_for_this_n
            best_n_for_overall_summary = current_n_value
            best_n_detailed_results_for_csv = temp_detailed_results_for_this_n_run # Save details of N
            best_n_confusion_totals_for_csv = current_n_run_confusion_totals.copy()
            best_n_stats_for_csv = current_n_run_stats.copy() # Save stats of N

    print(f"\n--- Baseline 2 (N-th Choice) Overall Best N Summary ---")
    if best_n_for_overall_summary == -1:
        print("No valid N-th option evaluation could be performed (e.g. no items with options).")
        return

    print(f"The best performance was achieved by always picking the {best_n_for_overall_summary}-th option.")
    print(f"Highest average score: {highest_avg_score_overall:.3f} / 2")

    # Save CSV for the best N
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    best_n_dir_name = f"result_{timestamp}_baseline2_best_n{best_n_for_overall_summary}_{dataset_name_for_path}"
    results_dir_best_n = result_dir / best_n_dir_name
    results_dir_best_n.mkdir(parents=True, exist_ok=True)
    results_csv_best_n = results_dir_best_n / "evaluation_results.csv"

    with open(results_csv_best_n,"w",newline="",encoding="utf-8-sig") as f_csv_best:
        writer = csv.DictWriter(f_csv_best, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_n_detailed_results_for_csv)

        total_row_best_n = {k:"" for k in fieldnames}
        total_row_best_n["identifier"]=f"TOTAL (Best N={best_n_for_overall_summary})"
        total_row_best_n.update(best_n_confusion_totals_for_csv)
        writer.writerow(total_row_best_n)

    print(f"\n--- Detailed Summary for Best N ({best_n_for_overall_summary}-th option choice) --- ")
    final_valid_items_best_n = best_n_stats_for_csv["processed"]
    final_max_score_best_n = final_valid_items_best_n * 2 if final_valid_items_best_n > 0 else 1
    final_avg_score_best_n = best_n_stats_for_csv["score_sum"] / max(final_valid_items_best_n, 1)
    final_percent_score_best_n = (best_n_stats_for_csv["score_sum"] / max(final_max_score_best_n, 1)) * 100

    print(f"Processed: {best_n_stats_for_csv['processed']}")
    print(f"Items where {best_n_for_overall_summary}-th option was not available: {best_n_stats_for_csv.get('items_without_nth_option', 'N/A')}")
    print(f"Average score: {final_avg_score_best_n:.3f} / 2  ({final_percent_score_best_n:.2f}%)")
    print(f"Gold  🥇: {best_n_stats_for_csv['gold']}  | Silver 🥈: {best_n_stats_for_csv['silver']}  | Neg ❌: {best_n_stats_for_csv['negative']}")
    print(f"Invalid key (pred not in options): {best_n_stats_for_csv['invalid_key']}  | Invalid resp (pred is None): {best_n_stats_for_csv['invalid_resp']}")
    print("Confusion totals (for Best N):")
    print(f"               | Predicted Positive | Predicted Negative |")
    print(f"---------------|--------------------|--------------------|")
    print(f"Actual Gold    | {best_n_confusion_totals_for_csv.get('gold_answer',0):<18} | {best_n_confusion_totals_for_csv.get('gold_negative',0):<18} |")
    print(f"Actual Silver  | {best_n_confusion_totals_for_csv.get('silver_answer',0):<18} | {best_n_confusion_totals_for_csv.get('silver_negative',0):<18} |")
    print(f"Actual Negative| {best_n_confusion_totals_for_csv.get('negative_answer',0):<18} | {best_n_confusion_totals_for_csv.get('negative_negative',0):<18} |")
    print(f"\nDetailed CSV for Best N ({best_n_for_overall_summary}-th option) → {results_csv_best_n}")

    print(f"\nDetailed results saved in directory: {results_dir_best_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run cited paragraph prediction benchmark test using DxD-Lab/PANORAMA-PI4PC-Bench dataset from Hugging Face.')

    parser.add_argument('--provider', type=str,
                       choices=['openai', 'anthropic', 'google'],
                       help='LLM provider (openai, anthropic, or google). Required if not in baseline mode.')
    parser.add_argument('--model', type=str,
                       help='Model name (e.g., gpt-4o, claude-3-opus-20240229, gemini-1.5-flash-latest). Required if not in baseline mode.')
    parser.add_argument('--prompt_mode', type=str, default='zero-shot',
                    choices=['zero-shot', 'cot', 'cot_base'],
                    help="Prompt style: 'zero-shot' (default) or 'cot' (Chain-of-Thought). Used in LLM mode.")

    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument('--baseline-1', action='store_true', help='Run baseline 1 mode (random choice).')
    baseline_group.add_argument('--baseline-2', action='store_true', help='Run baseline 2 mode (n-th choice).')

    parser.add_argument('--baseline-1-runs', type=int, default=20,
                        help='Number of times to run Baseline 1 evaluation (default: 20)')

    args = parser.parse_args()

    if not args.baseline_1 and not args.baseline_2: # LLM Mode
        if not args.provider or not args.model:
            parser.error("--provider and --model are required when not in baseline mode.")
    
    try:
        import pandas
        import pyarrow
        import datasets
    except ImportError:
        print("Error: 'pandas', 'pyarrow', and 'datasets' libraries are required.")
        print("Please install them using: pip install pandas pyarrow datasets")
        sys.exit(1)

    try:
        if args.baseline_1:
            run_baseline_1_evaluation(args)
        elif args.baseline_2:
            run_baseline_2_evaluation(args)
        else:
            run_llm_evaluation(args)

    except Exception as e:
        print(f"\nCritical error in main execution: {str(e)}")
        print(traceback.format_exc())
