# example(zero-shot): python benchmarks/noc4pc/inference.py --provider google --model gemini-1.5-flash-latest --prompt_mode zero-shot
# example(cot): python benchmarks/noc4pc/inference.py --provider openai --model gpt-4o --prompt_mode cot

# example(baseline): python benchmarks/noc4pc/inference.py --baseline-1
# example(baseline-2): python benchmarks/noc4pc/inference.py --baseline-2

import json
import os
import argparse
import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Literal
from datetime import datetime
from dotenv import load_dotenv
import traceback
import time
import re
import csv
import sys # For exit
import random
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from pydantic import BaseModel, Field
from datasets import load_dataset

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

warnings.filterwarnings(action='ignore')

DATASET_NAME = "DxD-Lab/PANORAMA-NOC4PC-Bench"

class ZeroShotRejectionAnswer(BaseModel):
    code: Literal["ALLOW", "102", "103"] = Field(description="The rejection code (ALLOW, 102, or 103).")

class CoTRejectionAnswer(BaseModel):
    reason: str = Field(description="Office Action style reasoning for the decision.")
    code: Literal["ALLOW", "102", "103"] = Field(description="The rejection code based on the reasoning.")


def get_chat_model(provider: str, model_name: str, prompt_mode: str):
    if provider.lower() == "openai":
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
    elif provider.lower() == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        anthropic_tools = [{
            "name": "decision",
            "description": "Return the decision code and optional reasoning",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code":   {"type": "string", "enum": ["ALLOW","102","103"]},
                    "reason": {"type": "string"}
                },
                "required": ["code"] if prompt_mode=="zero-shot" else ["code","reason"]
            }
        }]
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=1024,
            tools=anthropic_tools,
            tool_choice={"type":"tool","name":"decision"}
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
        if prompt_mode == "cot":
            structured_llm = chat_model_base.with_structured_output(CoTRejectionAnswer)
        if prompt_mode == "cot_base":
            structured_llm = chat_model_base.with_structured_output(CoTRejectionAnswer)
        else: # zero-shot
            structured_llm = chat_model_base.with_structured_output(ZeroShotRejectionAnswer)

        return structured_llm
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_rejection_prompt(benchmark_data: Dict[str, Any], prompt_mode: str = "zero-shot") -> str:
    """Rejection 벤치마크 데이터로부터 LLM 프롬프트를 생성합니다 (Reason 우선, 인용 발명 claim 포함)."""
    app_num = benchmark_data.get("application_number", "N/A")
    claim_num = benchmark_data.get("claim_number", "N/A")
    context_raw = benchmark_data.get("context", {})
    if isinstance(context_raw, str):
        try:
            context = json.loads(context_raw)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode context JSON for app {app_num}, claim {claim_num}. Using raw string or empty dict.")
            context = {}
    else:
        context = context_raw

    prior_art_raw = benchmark_data.get("prior_art_specifications", [])
    if isinstance(prior_art_raw, str):
        try:
            prior_art = json.loads(prior_art_raw)
            if not isinstance(prior_art, list): prior_art = []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode prior_art_specifications JSON for app {app_num}, claim {claim_num}. Using empty list.")
            prior_art = []
    else:
        prior_art = prior_art_raw if isinstance(prior_art_raw, list) else []


    target_title = context.get("title", "N/A")
    target_abstract = context.get("abstract", "N/A")
    target_claims = context.get("claims", [])

    target_claim_text = "N/A"
    try:
        claim_idx = int(claim_num) - 1
        if 0 <= claim_idx < len(target_claims):
            claim_item = target_claims[claim_idx]
            if isinstance(claim_item, dict) and 'claim_text' in claim_item:
                 target_claim_text = str(claim_item['claim_text'])
            elif isinstance(claim_item, str):
                 target_claim_text = claim_item
            else:
                 target_claim_text = str(claim_item) # Fallback
        else:
             # print(f"Warning: Claim number {claim_num} out of bounds for app {app_num}")
             pass
    except (ValueError, TypeError):
        # print(f"Warning: Invalid claim number {claim_num} for app {app_num}")
        pass

    prompt = f"""
You are an expert AI acting as a U.S. Patent Examiner.
Your task is to analyze **Target Claim {claim_num}** of the **Target Patent Application** in view of the provided **Prior Art Specifications**.

Determine if **Target Claim {claim_num}** is allowable or should be rejected under 35 U.S.C. § 102 (lack of novelty) or 35 U.S.C. § 103 (obviousness).

**Target Patent Application Information:**
*   Application Number: {app_num}
*   Target Claim Number: {claim_num}
*   Title: {target_title}
*   Abstract: {target_abstract}
*   Target Claim {claim_num} Text to Analyze:    ```
    {target_claim_text}    ```

**Prior Art Specifications (Cited as Basis for Potential Rejection):**
The following prior art documents and specific paragraphs were cited as potentially relevant for the rejection of the target claim. Analyze the target claim against the information presented in these specific paragraphs **and the claims** of the prior art.
"""
    if not prior_art:
        prompt += "No prior art specifications provided for analysis.\n"
    else:
        for i, spec in enumerate(prior_art):
            prompt += f"\n--- Prior Art #{i+1} ---\n"
            prompt += f"*   Patent ID: {spec.get('patent_id', 'N/A')}\n"
            prompt += f"*   Title: {spec.get('title', 'N/A')}\n"
            prompt += f"*   Abstract: {spec.get('abstract', 'N/A')}\n"

            prior_art_claims = spec.get('claims', [])
            prompt += f"*   Claims:\n"
            if not prior_art_claims:
                prompt += "    (No claims provided for this prior art)\n"
            else:
                for idx, claim_content in enumerate(prior_art_claims):
                     claim_text_display = str(claim_content)
                     if isinstance(claim_content, dict) and 'claim_text' in claim_content:
                          claim_text_display = str(claim_content['claim_text'])

                     claim_lines = claim_text_display.splitlines()
                     if claim_lines:
                          prompt += f"    {idx+1}. {claim_lines[0]}\n"
                          for line in claim_lines[1:]:
                               prompt += f"       {line}\n"
                     else:
                          prompt += f"    {idx+1}.\n"

            prompt += f"*   Cited Paragraphs from Specification (Basis for Analysis):\n"
            paragraphs = spec.get('paragraphs', [])
            if not paragraphs:
                prompt += "    (No specific paragraphs cited for this prior art)\n"
            else:
                for para in paragraphs:
                    key_display = para.get("key", "N/A")
                    content_display = para.get("content", "")
                    content_lines = content_display.splitlines()
                    if content_lines:
                         prompt += f'    [{key_display}]: {content_lines[0]}\n'
                         for line in content_lines[1:]:
                              prompt += f'          {line}\n'
                    else:
                         prompt += f'    [{key_display}]:\n'

    if prompt_mode == "cot":
        prompt += f"""
**Analysis Task and Response Instructions:**

Perform your internal reasoning first, then draft the *Office‑Action‑style* text.

---

### INTERNAL REASONING (not shown to applicant)
1. Apply the Broadest-Reasonable-Interpretation (BRI) to Claim {claim_num}; chart every limitation [L1]-[Ln].
   1-a. *Statutory check* - confirm Claim{claim_num} fits a statutory class (process, machine, manufacture, composition).
   1-b. *Limitation numbering*-  break the claim into [L1]-[Ln] and record in "element : function / relationship" form.
   1-c. *Key-feature flag* - mark limitations asserted (or apparent) as novel / non-obvious.
2. Compare each limitation to the teachings (claims + cited paragraphs) of every prior-art reference.
3. Decide:
   • §102 anticipation if a single reference explicitly, implicitly, or inherently discloses each and every limitation.
        Under BRI, interpret broadly: functional equivalence or conventional components (processors, databases, modules, memory, standard network elements, known protocols, etc.) count as implicit disclosures.
    • §103 obviousness if any of the following apply:
        (a) A primary reference discloses at least 70% of the limitations explicitly or implicitly, and remaining limitations constitute routine modifications, predictable optimizations (e.g., efficiency, speed, cost reduction, miniaturization), or standard practices known to a person of ordinary skill in the field.
        (b) A combination of references collectively covers all limitations and demonstrates a clear, implicit or explicit KSR rationale, such as addressing the same technical problem, improving performance, enhancing usability, or following common industry practices.
        (c) The limitations not explicitly disclosed are obvious through common general knowledge or widely recognized industry standards or textbooks in the field.
    • ALLOW Only if:
    No single reference or combination of references, even considering implicit disclosures and general knowledge, discloses or renders obvious specific, detailed implementation aspects (unique structures, algorithmic specifics, or non-trivial process steps), AND
    No reasonable motivation or rationale (performance improvement, standard practice, or known solution) can be objectively articulated to bridge these gaps.

---

### DRAFT OA LANGUAGE (will be revealed)
Write the *reason* paragraph exactly like an Office Action:

* Start: **"Regarding Claim {claim_num}, ..."**
* Use examiner diction:
  - "Reference X (Col. Y, lines Z) discloses ..."
  - "Therefore, Claim {claim_num} is rejected under 35 U.S.C. § 102(a) as being anticipated by Reference X."
  - or "It would have been obvious to one of ordinary skill to modify X with Y (same field, predictable results) ... ⇒ § 103 rejection."
  - or "The cited references do not teach or render obvious limitation [Lk] ... Claim {claim_num} is allowable."

* If § 103, list **all** references in the combination (e.g., "in view of Smith '123").
* Cite at least one column-line or paragraph for each matched limitation.
* Keep length ≤ 200 words.

---

### OUTPUT (JSON only)
Always write the "reason" first and then write the "answer".
Return exactly one JSON object:

```json
{{
  "reason": "<OA-style paragraph above>",
  "code": "102" | "103" | "ALLOW"
}}
```
"""
    if prompt_mode == "cot_base":
        prompt += f"""
**Select Conclusion Code**
Choose one: "ALLOW", "102", or "103".
    *   `"ALLOW"`: If your reasoning concluded the claim is novel and non-obvious over the cited art.
    *   `"102"` (Rejected - Novelty): If your reasoning concluded the claim is anticipated by a single cited reference.
    *   `"103"` (Rejected - Obviousness): If your reasoning concluded the claim is obvious over the cited art.

### OUTPUT (JSON only)
Think through the steps required to evaluate this, craft the supporting rationale accordingly, and then deliver your answer based on that rationale.
Always write the "reason" first and then write the "answer".
Return exactly one JSON object:

```json
{{
  "reason": "...",
  "code": "102" | "103" | "ALLOW"
}}
```
"""

    else: # ---------- zero‑shot (default) ----------
        prompt += f"""
**Select Conclusion Code**
Choose one: "ALLOW", "102", or "103".
    *   `"ALLOW"`: If your reasoning concluded the claim is novel and non-obvious over the cited art.
    *   `"102"` (Rejected - Novelty): If your reasoning concluded the claim is anticipated by a single cited reference.
    *   `"103"` (Rejected - Obviousness): If your reasoning concluded the claim is obvious over the cited art.

**Answer Format (JSON only)**
Return ONLY this JSON object - DO NOT INCLUDE ANY REASON IN THE ANSWER.

```json
{{"code": "102"}}
```
"""
    return prompt


def process_llm_response_rejection(response_data: Union[dict, BaseModel]) -> Tuple[str | None, str | None]:
    """Extracts 'code' and 'reason' from the LLM response (Pydantic model or dict)."""
    code = None
    reason = None
    try:
        if isinstance(response_data, BaseModel):
            code = response_data.code
            if hasattr(response_data, 'reason'):
                reason = response_data.reason
        elif isinstance(response_data, dict):
            code = response_data.get("code")
            reason = response_data.get("reason")
        else:
            print(f"Warning: Unexpected response type for process_llm_response_rejection: {type(response_data)}")
            try:
                code_match = re.search(r'"code"\s*:\s*"(\w+)"', str(response_data))
                reason_match = re.search(r'"reason"\s*:\s*"(.*?)?"', str(response_data), re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip().upper()
                if reason_match:
                    reason = reason_match.group(1).strip()
            except Exception:
                pass

        if code is not None:
            code = str(code).strip().upper()
            if code not in {"ALLOW", "102", "103"}:
                print(f"Warning: Invalid code '{code}' extracted. Setting to None.")
                code = None
        if reason is not None:
            reason = str(reason).strip()

        return code, reason

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not extract code/reason from response: {response_data}. Error: {e}")
        return None, None

def evaluate_rejection_prediction(predicted_code: str | None, gold_code: str | None) -> bool:
    """예측된 코드와 정답 코드가 일치하는지 평가합니다."""
    if predicted_code is None or gold_code is None:
        return False
    return predicted_code.upper() == gold_code.upper()

def run_baseline_1_evaluation(baseline_1_runs: int = 20):
    """
    baseline 1: random selection (among 102, 103, ALLOW)
    calculate average performance by running multiple times
    """
    print(f"Starting Baseline 1 (Random Selection) evaluation with {baseline_1_runs} runs")
    print(f"Loading benchmark data from Hugging Face dataset {DATASET_NAME} (test split)")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
        total_items = len(benchmark_data_list)
        print(f"Loaded {total_items} items from Hugging Face dataset.")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print(traceback.format_exc())
        return

    if not benchmark_data_list:
        print(f"Error: No data loaded from dataset")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    results_dir = result_dir / f"result_{timestamp}_baseline1_{dataset_name_for_path}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    stats_file = results_dir / "baseline1_stats.csv"
    

    f1_macro_scores = []
    precisions = []
    recalls = []

    possible_codes = ["102", "103", "ALLOW"]
    label_map = {"102": 0, "103": 1, "ALLOW": 2}

    for run in range(1, baseline_1_runs + 1):
        run_result_file = runs_dir / f"run_{run}_results.csv"
        
        csv_header = ["identifier", "target_patent", "target_claim",
                    "gold_code", "predicted_code", "is_code_correct"]

        try:
            with open(run_result_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
            print(f"\nRun {run}/{baseline_1_runs}: Results CSV file initialized: {run_result_file}")
        except IOError as e:
            print(f"Error initializing CSV file {run_result_file}: {e}")
            continue

        total_processed = 0
        correct_predictions = 0
        
        y_true = []
        y_pred = []

        for index, benchmark_data in enumerate(benchmark_data_list):
            total_processed += 1
            app_num = benchmark_data.get("application_number", "N/A")
            claim_num = benchmark_data.get("claim_number", "N/A")
            identifier = f"rejection_app{app_num}_cl{claim_num}_{index}"

            if total_processed % 100 == 0 or total_processed == total_items:
                print(f"\rProcessing Run {run}/{baseline_1_runs}: {total_processed}/{total_items} ({((total_processed)/total_items)*100:.1f}%)", end="")

            result_row_dict = {
                "identifier": identifier,
                "target_patent": app_num,
                "target_claim": claim_num,
                "gold_code": "N/A", 
                "predicted_code": "N/A", 
                "is_code_correct": False
            }

            answer_raw = benchmark_data.get("answer", {})
            if isinstance(answer_raw, str):
                try:
                    gold_answer = json.loads(answer_raw)
                    if not isinstance(gold_answer, dict): gold_answer = {}
                except json.JSONDecodeError:
                    gold_answer = {}
            else:
                gold_answer = answer_raw if isinstance(answer_raw, dict) else {}

            gold_code = str(gold_answer.get("code")).strip().upper() if gold_answer.get("code") else None
            result_row_dict["gold_code"] = gold_code if gold_code else "N/A"

            if gold_code and gold_code in possible_codes:
                predicted_code = random.choice(possible_codes)
                result_row_dict["predicted_code"] = predicted_code
                
                is_correct = evaluate_rejection_prediction(predicted_code, gold_code)
                result_row_dict["is_code_correct"] = is_correct
                
                if is_correct:
                    correct_predictions += 1
                
                y_true.append(label_map[gold_code])
                y_pred.append(label_map[predicted_code])

            try:
                with open(run_result_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    writer.writerow(result_row_dict)
            except IOError as e:
                print(f"\n  [!] Error writing row {index} to CSV {run_result_file}: {e}")

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1
            precision_macro = precision
            recall_macro = recall
        else:
            f1_macro = 0.0
            precision_macro = 0.0
            recall_macro = 0.0
            print(f"\nWarning: Run {run} - Not enough distinct classes to calculate metrics properly.")
        
        f1_macro_scores.append(f1_macro)
        precisions.append(precision_macro)
        recalls.append(recall_macro)
        
        valid_items = len(y_true)
        accuracy = (correct_predictions / valid_items) * 100 if valid_items > 0 else 0.0
        
        print(f"\nRun {run}/{baseline_1_runs}:")
        print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_items})")
        print(f"  Macro F1 Score: {f1_macro:.4f}")
        print(f"  Macro Precision: {precision_macro:.4f}")
        print(f"  Macro Recall: {recall_macro:.4f}")

    if f1_macro_scores:
        min_f1 = min(f1_macro_scores)
        max_f1 = max(f1_macro_scores)
        mean_f1 = np.mean(f1_macro_scores)
        std_f1 = np.std(f1_macro_scores)
        
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        
        print("\n--- Baseline 1 (Random Selection) Summary ---")
        print(f"Number of runs: {baseline_1_runs}")
        print(f"Minimum Macro F1: {min_f1:.4f}")
        print(f"Maximum Macro F1: {max_f1:.4f}")
        print(f"Average Macro F1: {mean_f1:.4f}")
        print(f"Standard deviation: {std_f1:.4f}")
        print(f"Average Macro Precision: {mean_precision:.4f}")
        print(f"Average Macro Recall: {mean_recall:.4f}")
        
        try:
            with open(stats_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["run", "macro_f1", "macro_precision", "macro_recall"])
                for i, (f1, prec, rec) in enumerate(zip(f1_macro_scores, precisions, recalls), 1):
                    writer.writerow([i, f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}"])
                writer.writerow([])
                writer.writerow(["stat", "value"])
                writer.writerow(["min_f1", f"{min_f1:.4f}"])
                writer.writerow(["max_f1", f"{max_f1:.4f}"])
                writer.writerow(["mean_f1", f"{mean_f1:.4f}"])
                writer.writerow(["std_f1", f"{std_f1:.4f}"])
                writer.writerow(["mean_precision", f"{mean_precision:.4f}"])
                writer.writerow(["mean_recall", f"{mean_recall:.4f}"])
            print(f"\nStatistics saved to: {stats_file}")
        except IOError as e:
            print(f"Error writing statistics to CSV {stats_file}: {e}")
    
    print(f"\nAll individual run results saved in directory: {runs_dir}")

def run_baseline_2_evaluation():
    """
    baseline 2: always the same selection (102, 103, ALLOW)
    compare performance for each option
    """
    print(f"Starting Baseline 2 (Fixed Selection) evaluation")
    print(f"Loading benchmark data from Hugging Face dataset {DATASET_NAME} (test split)")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
        total_items = len(benchmark_data_list)
        print(f"Loaded {total_items} items from Hugging Face dataset.")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print(traceback.format_exc())
        return

    if not benchmark_data_list:
        print(f"Error: No data loaded from dataset")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    results_dir = result_dir / f"result_{timestamp}_baseline2_{dataset_name_for_path}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_codes = ["102", "103", "ALLOW"]
    results = {}
    f1_results = {}
    precision_results = {}
    recall_results = {}
    
    label_map = {"102": 0, "103": 1, "ALLOW": 2}

    for fixed_code in fixed_codes:
        result_file = results_dir / f"baseline2_{fixed_code}_results.csv"
        
        csv_header = ["identifier", "target_patent", "target_claim",
                    "gold_code", "predicted_code", "is_code_correct"]

        try:
            with open(result_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
            print(f"\nFixed code '{fixed_code}': Results CSV file initialized: {result_file}")
        except IOError as e:
            print(f"Error initializing CSV file {result_file}: {e}")
            continue

        total_processed = 0
        correct_predictions = 0
        
        y_true = []
        y_pred = []

        for index, benchmark_data in enumerate(benchmark_data_list):
            total_processed += 1
            app_num = benchmark_data.get("application_number", "N/A")
            claim_num = benchmark_data.get("claim_number", "N/A")
            identifier = f"rejection_app{app_num}_cl{claim_num}_{index}"

            if total_processed % 100 == 0 or total_processed == total_items:
                print(f"\rProcessing Fixed '{fixed_code}': {total_processed}/{total_items} ({((total_processed)/total_items)*100:.1f}%)", end="")

            result_row_dict = {
                "identifier": identifier,
                "target_patent": app_num,
                "target_claim": claim_num,
                "gold_code": "N/A", 
                "predicted_code": fixed_code, 
                "is_code_correct": False
            }

            answer_raw = benchmark_data.get("answer", {})
            if isinstance(answer_raw, str):
                try:
                    gold_answer = json.loads(answer_raw)
                    if not isinstance(gold_answer, dict): gold_answer = {}
                except json.JSONDecodeError:
                    gold_answer = {}
            else:
                gold_answer = answer_raw if isinstance(answer_raw, dict) else {}

            gold_code = str(gold_answer.get("code")).strip().upper() if gold_answer.get("code") else None
            result_row_dict["gold_code"] = gold_code if gold_code else "N/A"

            if gold_code and gold_code in label_map:
                is_correct = evaluate_rejection_prediction(fixed_code, gold_code)
                result_row_dict["is_code_correct"] = is_correct
                
                if is_correct:
                    correct_predictions += 1
                
                y_true.append(label_map[gold_code])
                y_pred.append(label_map[fixed_code])

            try:
                with open(result_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    writer.writerow(result_row_dict)
            except IOError as e:
                print(f"\n  [!] Error writing row {index} to CSV {result_file}: {e}")

        valid_items = len(y_true)
        accuracy = (correct_predictions / valid_items) * 100 if valid_items > 0 else 0.0
        results[fixed_code] = accuracy
        
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            f1_results[fixed_code] = f1
            precision_results[fixed_code] = precision
            recall_results[fixed_code] = recall
        else:
            f1_results[fixed_code] = 0.0
            precision_results[fixed_code] = 0.0
            recall_results[fixed_code] = 0.0
            print(f"\nWarning: Fixed code '{fixed_code}' - Not enough distinct classes to calculate metrics properly.")
        
        print(f"\nFixed code '{fixed_code}':")
        print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_items})")
        print(f"  Macro F1 Score: {f1_results[fixed_code]:.4f}")
        print(f"  Macro Precision: {precision_results[fixed_code]:.4f}")
        print(f"  Macro Recall: {recall_results[fixed_code]:.4f}")

    if f1_results:
        best_code = max(f1_results, key=f1_results.get)
        best_f1 = f1_results[best_code]
        best_precision = precision_results[best_code]
        best_recall = recall_results[best_code]
        best_accuracy = results[best_code]
        
        print("\n--- Baseline 2 (Fixed Selection) Summary ---")
        for code in fixed_codes:
            print(f"Fixed code '{code}':")
            print(f"  Accuracy: {results[code]:.2f}%")
            print(f"  Macro F1: {f1_results[code]:.4f}")
            print(f"  Macro Precision: {precision_results[code]:.4f}")
            print(f"  Macro Recall: {recall_results[code]:.4f}")
        
        print(f"\nBest fixed code (by Macro F1): '{best_code}'")
        print(f"  Macro F1: {best_f1:.4f}")
        print(f"  Accuracy: {best_accuracy:.2f}%")
        print(f"  Macro Precision: {best_precision:.4f}")
        print(f"  Macro Recall: {best_recall:.4f}")
        
        summary_file = results_dir / "baseline2_summary.csv"
        try:
            with open(summary_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["fixed_code", "accuracy", "macro_f1", "macro_precision", "macro_recall"])
                for code in fixed_codes:
                    writer.writerow([code, f"{results[code]:.2f}", f"{f1_results[code]:.4f}", 
                                    f"{precision_results[code]:.4f}", f"{recall_results[code]:.4f}"])
                writer.writerow([])
                writer.writerow(["best_code", "best_accuracy", "best_f1", "best_precision", "best_recall"])
                writer.writerow([best_code, f"{best_accuracy:.2f}", f"{best_f1:.4f}", 
                                f"{best_precision:.4f}", f"{best_recall:.4f}"])
            print(f"\nSummary saved to: {summary_file}")
        except IOError as e:
            print(f"Error writing summary to CSV {summary_file}: {e}")
    
    print(f"\nAll results saved in directory: {results_dir}")

def main(provider: str, model_name: str, prompt_mode: str):
    print(f"Starting Rejection benchmark test with {provider} - {model_name}")
    print(f"Loading benchmark data from Hugging Face dataset {DATASET_NAME} (test split)")

    try:
        ds = load_dataset(DATASET_NAME, split="test")
        df = ds.to_pandas()
        benchmark_data_list = df.to_dict("records")
        total_items = len(benchmark_data_list)
        print(f"Loaded {total_items} items from Hugging Face dataset.")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print(traceback.format_exc())
        return

    if not benchmark_data_list:
        print(f"Error: No data loaded from dataset")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_for_path = DATASET_NAME.replace("/", "_")
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    results_dir = result_dir / f"result_{timestamp}_{provider}_{model_name}_{prompt_mode}_{dataset_name_for_path}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "evaluation_results.csv"
    error_log_file = results_dir / "error_log.txt"

    csv_header = ["identifier", "target_patent", "target_claim",
                  "gold_code", "predicted_code", "is_code_correct",
                  "gold_reason", "predicted_reason", "llm_raw_response", "error"]

    try:
        with open(results_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
        print(f"Results CSV file initialized: {results_file}")
    except IOError as e:
        print(f"Error initializing CSV file {results_file}: {e}"); return


    all_gold_codes = []
    all_predicted_codes = []


    total_processed = 0
    errors = 0
    correct_code_predictions = 0

    chat = get_chat_model(provider, model_name, prompt_mode)

    for index, benchmark_data in enumerate(benchmark_data_list):
        total_processed += 1
        app_num = benchmark_data.get("application_number", "N/A")
        claim_num = benchmark_data.get("claim_number", "N/A")
        identifier = f"rejection_app{app_num}_cl{claim_num}_{index}"

        print(f"\rProcessing: {total_processed}/{total_items} ({((total_processed)/total_items)*100:.1f}%) - {identifier}", end="")

        llm_response_object = None
        llm_raw_response_str = ""
        predicted_code = None
        predicted_reason = None
        is_code_correct = False
        error_message = ""
        success = False

        result_row_dict = {
            "identifier": identifier,
            "target_patent": app_num,
            "target_claim": claim_num,
            "gold_code": "N/A", "predicted_code": "N/A", "is_code_correct": False,
            "gold_reason": "N/A", "predicted_reason": "N/A", "llm_raw_response": "", "error": ""
        }

        try:
            answer_raw = benchmark_data.get("answer", {})
            if isinstance(answer_raw, str):
                try:
                    gold_answer = json.loads(answer_raw)
                    if not isinstance(gold_answer, dict): gold_answer = {}
                except json.JSONDecodeError:
                    print(f"\nWarning: Could not decode answer JSON for {identifier}. Treating as empty.")
                    gold_answer = {}
            else:
                gold_answer = answer_raw if isinstance(answer_raw, dict) else {}

            gold_code = str(gold_answer.get("code")).strip().upper() if gold_answer.get("code") else None
            gold_reason = str(gold_answer.get("reason")).strip() if gold_answer.get("reason") else None
            result_row_dict["gold_code"] = gold_code if gold_code else "N/A"
            result_row_dict["gold_reason"] = gold_reason if gold_reason else "N/A"


            if not gold_code:
                 print(f"\nWarning: Missing gold 'code' for {identifier}. Skipping evaluation.")
                 error_message = "Missing gold code in benchmark data"
                 result_row_dict["error"] = error_message
                 errors += 1
            else:
                prompt = create_rejection_prompt(benchmark_data, prompt_mode)

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        system_message = SystemMessage(content="You are an expert patent examiner predicting claim rejection. Respond in the requested JSON format.")
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
                                llm_response_object = {"code": None, "reason": None}
                                llm_raw_response_str = response.content.strip()
                                print(f"\nWarning: Anthropic response did not contain tool calls: {llm_raw_response_str}")
                        else:  # openai
                            response = chat([system_message, human_message])
                            llm_raw_response_str = response.content.strip()
                            try:
                                llm_response_object = json.loads(llm_raw_response_str)
                            except json.JSONDecodeError:
                                print(f"\nWarning: Could not parse OpenAI response as JSON: {llm_raw_response_str}")
                                llm_response_object = {"code": None, "reason": None}


                        predicted_code, predicted_reason = process_llm_response_rejection(llm_response_object)
                        error_message = ""
                        success = True
                        break
                    except Exception as e:
                        error_message = f"LLM call failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}"
                        llm_raw_response_str = f"Error: {error_message}"
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            time.sleep(wait_time)

                if success:
                    if predicted_code is not None:
                        is_code_correct = evaluate_rejection_prediction(predicted_code, gold_code)
                        if is_code_correct:
                            correct_code_predictions += 1
                    else:
                         error_message = "LLM response parsing failed (Code could not be extracted)"
                         errors += 1
                else:
                    errors += 1

            result_row_dict["predicted_code"] = predicted_code if predicted_code else "N/A"
            result_row_dict["is_code_correct"] = is_code_correct if success and predicted_code is not None else False
            result_row_dict["predicted_reason"] = predicted_reason if predicted_reason else "N/A"
            result_row_dict["llm_raw_response"] = llm_raw_response_str
            result_row_dict["error"] = error_message

            all_gold_codes.append(gold_code)
            all_predicted_codes.append(predicted_code)


            try:
                with open(results_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                    writer.writerow(result_row_dict)
            except IOError as e:
                print(f"\n  [!] Error writing row {index} to CSV {results_file}: {e}")


        except Exception as e:
            error_msg = f"Error processing data for item {index} ({identifier}): {type(e).__name__}: {e}"
            print(f"\n  [!] {error_msg}")
            print(traceback.format_exc())
            errors += 1
            with open(error_log_file, 'a', encoding='utf-8') as f_err:
                 f_err.write(f"{datetime.now().isoformat()} - {error_msg}\n")
                 f_err.write(traceback.format_exc() + "\n\n")
            result_row_dict["error"] = error_msg



    print("\nCalculating and logging 3x3 confusion matrix for the entire dataset...")

    confusion_matrix = {
        "102_102": 0, "102_103": 0, "102_ALLOW": 0, "102_None": 0,
        "103_102": 0, "103_103": 0, "103_ALLOW": 0, "103_None": 0,
        "ALLOW_102": 0, "ALLOW_103": 0, "ALLOW_ALLOW": 0, "ALLOW_None": 0,
        "None_102": 0, "None_103": 0, "None_ALLOW": 0, "None_None": 0
    }
    categories = ["102", "103", "ALLOW", None]

    for gold, predicted in zip(all_gold_codes, all_predicted_codes):
        gold_cat = gold if gold in ["102", "103", "ALLOW"] else None
        pred_cat = predicted if predicted in ["102", "103", "ALLOW"] else None
        key = f"{gold_cat}_{pred_cat}"
        if key in confusion_matrix:
            confusion_matrix[key] += 1
        else:
            print(f"Warning: Unexpected confusion matrix key combination: Gold={gold}, Predicted={predicted}")


    print("\nConfusion Matrix (Gold \\ Predicted):")
    header = "         | {:^10} | {:^10} | {:^10} | {:^10} |".format("Pred 102", "Pred 103", "Pred ALLOW", "Pred None")
    print(header)
    print("-" * len(header))
    for gold_cat in ["102", "103", "ALLOW", None]:
        row = f"Gold {str(gold_cat):<5} |"
        for pred_cat in ["102", "103", "ALLOW", None]:
            key = f"{gold_cat}_{pred_cat}"
            count = confusion_matrix.get(key, 0)
            row += f" {count:<10} |"
        print(row)


    print("\n--- Rejection Benchmark Test Summary ---")
    print(f"Total Items Loaded: {total_items}")
    print(f"Total Items Processed: {total_processed}")
    print(f"Correct Code Predictions: {correct_code_predictions}")
    print(f"Processing Errors (Data/LLM/Parsing): {errors}")

    valid_evaluated_items = sum(1 for i in range(len(all_predicted_codes)) if all_predicted_codes[i] is not None and all_gold_codes[i] is not None)

    accuracy = (correct_code_predictions / valid_evaluated_items) * 100 if valid_evaluated_items > 0 else 0.0
    print(f"Code Prediction Accuracy (over successfully processed & evaluated items): {accuracy:.2f}% ({correct_code_predictions}/{valid_evaluated_items})")

    print(f"\nDetailed results saved to: {results_file}")
    if errors > 0 and os.path.exists(error_log_file) and os.path.getsize(error_log_file) > 0:
         print(f"Error details logged to: {error_log_file}")
    elif errors > 0:
         print(f"Processing errors occurred, check terminal output or log file for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rejection prediction benchmark test using DxD-Lab/PANORAMA-NOC4PC-Bench dataset from Hugging Face.')
    parser.add_argument('--provider', type=str, required=False,
                       choices=['openai', 'anthropic', 'google'],
                       help='LLM provider (openai, anthropic, or google)')
    parser.add_argument('--model', type=str, required=False,
                       help='Model name (e.g., gpt-4o, claude-3-opus-20240229, gemini-1.5-flash-latest)')
    parser.add_argument('--prompt_mode', type=str, default='zero-shot',
                        choices=['zero-shot', 'cot', 'cot_base'],
                        help="Prompt style: 'zero-shot' (default) or 'cot' (Chain-of-Thought)")

    parser.add_argument('--baseline-1', action='store_true',
                        help='Run Baseline 1: random selection from possible codes')
    parser.add_argument('--baseline-1-runs', type=int, default=20,
                        help='Number of runs for Baseline 1 (default: 20)')
    parser.add_argument('--baseline-2', action='store_true',
                        help='Run Baseline 2: fixed selection for each possible code')
    
    args = parser.parse_args()

    if args.baseline_1:
        run_baseline_1_evaluation(args.baseline_1_runs)
    elif args.baseline_2:
        run_baseline_2_evaluation()
    else:
        if not args.provider or not args.model:
            parser.error("--provider and --model are required when not using baseline modes")
        
        try:
            import pandas
            import pyarrow
            import datasets
        except ImportError:
            print("Error: 'pandas', 'pyarrow', and 'datasets' libraries are required.")
            print("Please install them using: pip install pandas pyarrow datasets")
            sys.exit(1)

        try:
            main(args.provider, args.model, args.prompt_mode)
        except Exception as e:
            print(f"\nCritical error in main execution: {str(e)}")
            print(traceback.format_exc())
