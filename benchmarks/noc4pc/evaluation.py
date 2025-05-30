# python benchmarks/noc4pc/evaluation.py <path_to_evaluation_results.csv>

import pandas as pd
import argparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback
from pathlib import Path
import time
import subprocess
import sys

similarity_libs_available = False
try:
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    from bleurt import score
    similarity_libs_available = True
except ImportError as e:
    print(f"Warning: Similarity calculation libraries not found ({e}). Attempting to install...")
    install_command = [
        sys.executable,
        "-m", "pip", "install",
        "sentence-transformers", "rouge-score", "bert-score",
        "bleurt-score", "torch", "transformers", "scikit-learn", "pandas"
    ]
    try:
        result = subprocess.run(install_command, check=True, capture_output=True, text=True)
        print("Libraries installation successful.")
        try:
            from sentence_transformers import SentenceTransformer
            from rouge_score import rouge_scorer
            from bert_score import BERTScorer
            from bleurt import score
            similarity_libs_available = True
            print("Libraries imported successfully after installation.")
        except ImportError as inner_e:
            print(f"Error: Failed to import libraries even after installation: {inner_e}")
            print("Please check your Python environment and permissions.")
            similarity_libs_available = False

    except subprocess.CalledProcessError as install_err:
        print(f"Error: Failed to install libraries using pip.")
        print(f"Command failed: {' '.join(install_command)}")
        print(f"Pip stdout:\n{install_err.stdout}")
        print(f"Pip stderr:\n{install_err.stderr}")
        print("Please try installing the libraries manually:")
        print("pip install sentence-transformers rouge-score bert-score bleurt-score torch transformers scikit-learn pandas")
        similarity_libs_available = False
    except Exception as general_err:
        print(f"An unexpected error occurred during library installation: {general_err}")
        similarity_libs_available = False


def calculate_rejection_metrics(df, rejection_code):
    if 'error' in df.columns:
        valid_df = df[df['error'].isna()].copy()
    else:
        valid_df = df.copy()

    filtered_df = valid_df[valid_df['gold_code'].astype(str) == str(rejection_code)]
    
    total_items = len(filtered_df)
    if total_items == 0:
        return {
            'code': rejection_code,
            'total': 0,
            'correct': 0,
            'accuracy': 0,
            'custom_score': 0
        }
    
    correct = (filtered_df['predicted_code'].astype(str) == filtered_df['gold_code'].astype(str)).sum()
    accuracy = correct / total_items if total_items > 0 else 0
    
    custom_score = 0
    if 'model_score' in filtered_df.columns and 'max_score' in filtered_df.columns:
        total_model_score = filtered_df['model_score'].astype(float).clip(lower=0).fillna(0).sum()
        total_max_score = filtered_df['max_score'].astype(float).fillna(0).sum()
        custom_score = total_model_score / total_max_score * 100 if total_max_score > 0 else 0
    
    return {
        'code': rejection_code,
        'total': total_items,
        'correct': correct,
        'accuracy': accuracy,
        'custom_score': custom_score
    }

def calculate_f1_for_code(df, code):
    code_df = df[df['gold_code'].astype(str) == str(code)]
    
    if len(code_df) == 0:
        return 0.0
    
    y_true = code_df['gold_code']
    y_pred = code_df['predicted_code']
    
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    return f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)

def main(csv_filepath: str):
    print(f"Analyzing rejection results from: {csv_filepath}")
    start_time = time.time()

    input_path = Path(csv_filepath)
    output_filename = input_path.stem + "_with_rejection_analysis.csv"
    output_path = input_path.parent / output_filename

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print(traceback.format_exc())
        return

    total_items = len(df)
    if total_items == 0:
        print("CSV file is empty. No statistics to calculate.")
        return

    error_count = 0
    if 'error' in df.columns:
        error_count = df['error'].notna().sum()
    else:
        print("Warning: 'error' column not found. Assuming 0 errors.")

    valid_items_for_classification = total_items - error_count

    print(f"\n--- Overall Summary ---")
    print(f"Total Items Attempted: {total_items}")
    print(f"Processing Errors:     {error_count}")
    print(f"Valid Items for Classification Metrics: {valid_items_for_classification}")

    if valid_items_for_classification == 0:
        print("\nNo valid items to evaluate classification metrics.")
    else:
        if 'error' in df.columns:
            valid_df_class = df[df['error'].isna()].copy()
        else:
            valid_df_class = df.copy()

        if not all(col in valid_df_class.columns for col in ['gold_code', 'predicted_code']):
            print("\nError: Missing 'gold_code' or 'predicted_code' columns in the CSV.")
            print("Cannot calculate classification metrics.")
        else:
            valid_df_class['predicted_code'] = valid_df_class['predicted_code'].fillna('PREDICTION_FAILED').astype(str)
            valid_df_class['gold_code'] = valid_df_class['gold_code'].astype(str)

            y_true = valid_df_class['gold_code']
            y_pred = valid_df_class['predicted_code']

            labels = sorted(list(set(y_true) | set(y_pred) | {"ALLOW", "102", "103"}))
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
            report = classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4)
            try:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=[f'True_{l}' for l in labels], columns=[f'Pred_{l}' for l in labels])
            except Exception as e:
                print(f"\nWarning: Could not generate confusion matrix. Error: {e}")
                cm_df = None

            print(f"\n--- Classification Metrics (Based on {valid_items_for_classification} Valid Items) ---")
            print(f"1. Overall Accuracy: {accuracy:.4f} ({int(accuracy * valid_items_for_classification)} / {valid_items_for_classification})")
            print(f"2. Macro-F1 Score:   {macro_f1:.4f}")
            print("\n3. Classification Report:")
            print(report)
            if cm_df is not None:
                print("\n4. Confusion Matrix:")
                print(cm_df)
            else:
                print("\n4. Confusion Matrix: Not Available")
            
            rejection_codes = ["102", "103", "ALLOW"]
            metrics_results = []
            
            print("\n--- Rejection Code Specific Metrics ---")
            for code in rejection_codes:
                metrics = calculate_rejection_metrics(df, code)
                
                if metrics['total'] > 0:
                    macro_f1_score = calculate_f1_for_code(valid_df_class, code)
                    metrics['macro_f1_score'] = macro_f1_score
                else:
                    metrics['macro_f1_score'] = 0.0
                
                metrics_results.append(metrics)
                
                print(f"\n=== {code} Rejection Code Analysis ===")
                print(f"Total Items with Gold Code '{code}': {metrics['total']}")
                print(f"Correctly Predicted: {metrics['correct']} / {metrics['total']}")
                print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']} / {metrics['total']})")
                
                if metrics['total'] > 0:
                    print(f"Macro-F1 Score: {metrics['macro_f1_score']:.4f}")
                
                print(f"Custom Score: {metrics['custom_score']:.2f}%")
            
            print("\n=== Summary of Rejection Code Analysis ===")
            print("-" * 70)
            print(f"{'Rejection Code':<15} {'Total':<10} {'Correct':<10} {'Accuracy':<15} {'Macro-F1':<15}")
            print("-" * 70)
            for metrics in metrics_results:
                print(f"{metrics['code']:<15} {metrics['total']:<10} {metrics['correct']:<10} {metrics['accuracy']:.4f} {'':<8} {metrics['macro_f1_score']:.4f}")

            all_metrics = calculate_rejection_metrics(df, None)
            all_correct = (valid_df_class['predicted_code'] == valid_df_class['gold_code']).sum()
            all_accuracy = all_correct / len(valid_df_class) if len(valid_df_class) > 0 else 0
            
            print("-" * 70)
            print(f"{'OVERALL':<15} {valid_items_for_classification:<10} {all_correct:<10} {all_accuracy:.4f} {'':<8} {macro_f1:.4f}")
            print("-" * 70)

    print("\n--- Reason Similarity Calculation (Comparing 'gold_reason' and 'predicted_reason') ---")

    if not similarity_libs_available:
        print("Skipping similarity calculation because required libraries are missing or installation failed.")
        return

    if not all(col in df.columns for col in ['gold_reason', 'predicted_reason']):
        print("Error: Missing 'gold_reason' or 'predicted_reason' columns.")
        print("Cannot calculate reason similarity. Ensure the CSV was generated from a CoT run.")
        return

    valid_df_sim = df[
        df['error'].isna() &
        df['gold_reason'].notna() &
        df['predicted_reason'].notna()
    ].copy()

    valid_df_sim['gold_reason'] = valid_df_sim['gold_reason'].fillna('').astype(str)
    valid_df_sim['predicted_reason'] = valid_df_sim['predicted_reason'].fillna('').astype(str)

    valid_df_sim = valid_df_sim[
        (valid_df_sim['gold_reason'].str.strip() != '') &
        (valid_df_sim['predicted_reason'].str.strip() != '')
    ]

    valid_df_sim = valid_df_sim[valid_df_sim['gold_code'].astype(str) != 'ALLOW']

    valid_df_sim_102 = valid_df_sim[valid_df_sim['gold_code'].astype(str) == '102']
    valid_df_sim_103 = valid_df_sim[valid_df_sim['gold_code'].astype(str) == '103']

    valid_items_for_similarity = len(valid_df_sim)
    valid_items_for_similarity_102 = len(valid_df_sim_102)
    valid_items_for_similarity_103 = len(valid_df_sim_103)
    skipped_for_similarity = (total_items - error_count) - valid_items_for_similarity

    print(f"Total Valid Items (No Errors): {total_items - error_count}")
    print(f"Items with Valid gold_reason & predicted_reason (excluding ALLOW): {valid_items_for_similarity}")
    print(f"   - Items with gold_code 102: {valid_items_for_similarity_102}")
    print(f"   - Items with gold_code 103: {valid_items_for_similarity_103}")
    print(f"Items Skipped for Similarity (Errors, Missing Reasons, or ALLOW): {error_count + skipped_for_similarity}")

    if valid_items_for_similarity == 0:
        print("\nNo valid items found for similarity calculation.")
        return

    print("\nLoading similarity models...")
    try:
        cs_model = SentenceTransformer('all-MiniLM-L6-v2')
        rouge_calc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type='bert-base-uncased')

        script_dir = Path(__file__).resolve().parent
        bleurt_checkpoint_path = script_dir / 'bleurt' / 'bleurt' / 'test_checkpoint'

        if not bleurt_checkpoint_path.is_dir():
            print(f"Error: BLEURT checkpoint directory not found at calculated path: {bleurt_checkpoint_path}")
            print("Please ensure the 'bleurt/bleurt/test_checkpoint' directory exists relative to the script location and contains necessary files (e.g., 'bert_config.json').")
            print("Cannot proceed with similarity calculation due to missing BLEURT checkpoint.")
            return

        scorer = score.BleurtScorer(str(bleurt_checkpoint_path))

        print("Similarity models loaded successfully.")
    except AssertionError as ae:
         if "Could not find BLEURT checkpoint" in str(ae):
              print(f"Error: BLEURT loading failed unexpectedly. Checkpoint path used: {bleurt_checkpoint_path}")
              print("Ensure the directory contains valid BLEURT checkpoint files.")
         else:
              print(f"\nAssertionError loading similarity models: {ae}")
         print(traceback.format_exc())
         print("Cannot proceed with similarity calculation.")
         return
    except Exception as e:
        print(f"\nError loading similarity models: {e}")
        print(traceback.format_exc())
        print("Cannot proceed with similarity calculation.")
        return

    score_results = {
        'all': {
            'cosine_scores': [], 'rouge1_f1_scores': [], 'rouge2_f1_scores': [], 'rougeL_f1_scores': [],
            'bert_p_scores': [], 'bert_r_scores': [], 'bert_f1_scores': [], 'bleurt_scores': []
        },
        '102': {
            'cosine_scores': [], 'rouge1_f1_scores': [], 'rouge2_f1_scores': [], 'rougeL_f1_scores': [],
            'bert_p_scores': [], 'bert_r_scores': [], 'bert_f1_scores': [], 'bleurt_scores': []
        },
        '103': {
            'cosine_scores': [], 'rouge1_f1_scores': [], 'rouge2_f1_scores': [], 'rougeL_f1_scores': [],
            'bert_p_scores': [], 'bert_r_scores': [], 'bert_f1_scores': [], 'bleurt_scores': []
        }
    }

    print(f"\nCalculating similarity scores for {valid_items_for_similarity} items...")
    calculation_start_time = time.time()

    for index, row in valid_df_sim.iterrows():
        gold = row['gold_reason']
        pred = row['predicted_reason']
        gold_code = str(row['gold_code'])
        item_id = row.get('identifier', f'row_{index}')

        # 1. Cosine Similarity
        try:
            embeddings = cs_model.encode([gold, pred])
            cos_sim = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
            score_results['all']['cosine_scores'].append(cos_sim)
            if gold_code == '102':
                score_results['102']['cosine_scores'].append(cos_sim)
            elif gold_code == '103':
                score_results['103']['cosine_scores'].append(cos_sim)
        except Exception as e:
            print(f"\nWarning: Cosine Similarity failed for {item_id}. Error: {e}")
            score_results['all']['cosine_scores'].append(np.nan)
            if gold_code == '102':
                score_results['102']['cosine_scores'].append(np.nan)
            elif gold_code == '103':
                score_results['103']['cosine_scores'].append(np.nan)

        # 2. ROUGE
        try:
            rouge_results = rouge_calc.score(gold, pred)
            score_results['all']['rouge1_f1_scores'].append(rouge_results['rouge1'].fmeasure)
            score_results['all']['rouge2_f1_scores'].append(rouge_results['rouge2'].fmeasure)
            score_results['all']['rougeL_f1_scores'].append(rouge_results['rougeL'].fmeasure)

            if gold_code == '102':
                score_results['102']['rouge1_f1_scores'].append(rouge_results['rouge1'].fmeasure)
                score_results['102']['rouge2_f1_scores'].append(rouge_results['rouge2'].fmeasure)
                score_results['102']['rougeL_f1_scores'].append(rouge_results['rougeL'].fmeasure)
            elif gold_code == '103':
                score_results['103']['rouge1_f1_scores'].append(rouge_results['rouge1'].fmeasure)
                score_results['103']['rouge2_f1_scores'].append(rouge_results['rouge2'].fmeasure)
                score_results['103']['rougeL_f1_scores'].append(rouge_results['rougeL'].fmeasure)
        except Exception as e:
            print(f"\nWarning: ROUGE calculation failed for {item_id}. Error: {e}")
            score_results['all']['rouge1_f1_scores'].append(np.nan)
            score_results['all']['rouge2_f1_scores'].append(np.nan)
            score_results['all']['rougeL_f1_scores'].append(np.nan)

            if gold_code == '102':
                score_results['102']['rouge1_f1_scores'].append(np.nan)
                score_results['102']['rouge2_f1_scores'].append(np.nan)
                score_results['102']['rougeL_f1_scores'].append(np.nan)
            elif gold_code == '103':
                score_results['103']['rouge1_f1_scores'].append(np.nan)
                score_results['103']['rouge2_f1_scores'].append(np.nan)
                score_results['103']['rougeL_f1_scores'].append(np.nan)

        # 3. BERTScore
        try:
            P, R, F1 = bert_scorer.score([pred], [gold], verbose=False)
            score_results['all']['bert_p_scores'].append(P.item())
            score_results['all']['bert_r_scores'].append(R.item())
            score_results['all']['bert_f1_scores'].append(F1.item())
 
            if gold_code == '102':
                score_results['102']['bert_p_scores'].append(P.item())
                score_results['102']['bert_r_scores'].append(R.item())
                score_results['102']['bert_f1_scores'].append(F1.item())
            elif gold_code == '103':
                score_results['103']['bert_p_scores'].append(P.item())
                score_results['103']['bert_r_scores'].append(R.item())
                score_results['103']['bert_f1_scores'].append(F1.item())
        except Exception as e:
            print(f"\nWarning: BERTScore calculation failed for {item_id}. Error: {e}")
            score_results['all']['bert_p_scores'].append(np.nan)
            score_results['all']['bert_r_scores'].append(np.nan)
            score_results['all']['bert_f1_scores'].append(np.nan)
            
            if gold_code == '102':
                score_results['102']['bert_p_scores'].append(np.nan)
                score_results['102']['bert_r_scores'].append(np.nan)
                score_results['102']['bert_f1_scores'].append(np.nan)
            elif gold_code == '103':
                score_results['103']['bert_p_scores'].append(np.nan)
                score_results['103']['bert_r_scores'].append(np.nan)
                score_results['103']['bert_f1_scores'].append(np.nan)

        # 4. BLEURT
        try:
            bleurt_result = scorer.score(references=[gold], candidates=[pred])
            score_results['all']['bleurt_scores'].append(bleurt_result[0])
            
            if gold_code == '102':
                score_results['102']['bleurt_scores'].append(bleurt_result[0])
            elif gold_code == '103':
                score_results['103']['bleurt_scores'].append(bleurt_result[0])
        except Exception as e:
            print(f"\nWarning: BLEURT calculation failed for {item_id}. Error: {e}")
            score_results['all']['bleurt_scores'].append(np.nan)
            
            if gold_code == '102':
                score_results['102']['bleurt_scores'].append(np.nan)
            elif gold_code == '103':
                score_results['103']['bleurt_scores'].append(np.nan)

        processed_count = len(score_results['all']['cosine_scores'])
        if processed_count % 50 == 0 or processed_count == valid_items_for_similarity:
            elapsed = time.time() - calculation_start_time
            print(f"\r  Processed {processed_count}/{valid_items_for_similarity} items for similarity ({elapsed:.1f}s)...", end="")

    print("\nSimilarity calculation finished.")

    df['cosine_similarity'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['rouge1_f1'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['rouge2_f1'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['rougeL_f1'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['bertscore_precision'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['bertscore_recall'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['bertscore_f1'] = pd.Series(pd.Series(np.nan, index=df.index))
    df['bleurt_score'] = pd.Series(pd.Series(np.nan, index=df.index))
    
    df.loc[valid_df_sim.index, 'cosine_similarity'] = score_results['all']['cosine_scores']
    df.loc[valid_df_sim.index, 'rouge1_f1'] = score_results['all']['rouge1_f1_scores']
    df.loc[valid_df_sim.index, 'rouge2_f1'] = score_results['all']['rouge2_f1_scores'] 
    df.loc[valid_df_sim.index, 'rougeL_f1'] = score_results['all']['rougeL_f1_scores']
    df.loc[valid_df_sim.index, 'bertscore_precision'] = score_results['all']['bert_p_scores']
    df.loc[valid_df_sim.index, 'bertscore_recall'] = score_results['all']['bert_r_scores']
    df.loc[valid_df_sim.index, 'bertscore_f1'] = score_results['all']['bert_f1_scores']
    df.loc[valid_df_sim.index, 'bleurt_score'] = score_results['all']['bleurt_scores']

    score_columns = ['cosine_similarity', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1',
                     'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'bleurt_score']
    
    print(f"\n--- Average Similarity Scores for 102 & 103 Combined (Based on {valid_items_for_similarity} Valid Items) ---")
    avg_scores_all = {}
    for col, key in zip(score_columns, 
                      ['cosine_scores', 'rouge1_f1_scores', 'rouge2_f1_scores', 'rougeL_f1_scores',
                       'bert_p_scores', 'bert_r_scores', 'bert_f1_scores', 'bleurt_scores']):
        scores = score_results['all'][key]
        avg_score = np.nanmean(scores) if scores else np.nan
        avg_scores_all[col] = avg_score
        print(f"{col:<20}: {avg_score:.4f}")
    
    print(f"\n--- Average Similarity Scores for 102 (Based on {valid_items_for_similarity_102} Valid Items) ---")
    avg_scores_102 = {}
    for col, key in zip(score_columns,
                      ['cosine_scores', 'rouge1_f1_scores', 'rouge2_f1_scores', 'rougeL_f1_scores',
                       'bert_p_scores', 'bert_r_scores', 'bert_f1_scores', 'bleurt_scores']):
        scores = score_results['102'][key]
        avg_score = np.nanmean(scores) if scores else np.nan
        avg_scores_102[col] = avg_score
        print(f"{col:<20}: {avg_score:.4f}")
    
    print(f"\n--- Average Similarity Scores for 103 (Based on {valid_items_for_similarity_103} Valid Items) ---")
    avg_scores_103 = {}
    for col, key in zip(score_columns,
                      ['cosine_scores', 'rouge1_f1_scores', 'rouge2_f1_scores', 'rougeL_f1_scores',
                       'bert_p_scores', 'bert_r_scores', 'bert_f1_scores', 'bleurt_scores']):
        scores = score_results['103'][key]
        avg_score = np.nanmean(scores) if scores else np.nan
        avg_scores_103[col] = avg_score
        print(f"{col:<20}: {avg_score:.4f}")

    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")
        print(traceback.format_exc())

    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze rejection evaluation results CSV, calculate similarity, and report metrics.')
    parser.add_argument('csv_filepath', type=str,
                        help='Path to the evaluation_results.csv file for the rejection benchmark')

    args = parser.parse_args()

    main(args.csv_filepath)