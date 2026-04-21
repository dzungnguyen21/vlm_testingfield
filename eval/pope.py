import json
import logging
from typing import List, Dict, Union
import re
import argparse
import os
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_pope_dataset(cache_dir="~/POPE"):
    """
    Loads the POPE dataset from lmms-lab/POPE.
    Based on user configuration.
    """
    logger.info(f"Loading POPE dataset from lmms-lab/POPE")
    pope_dataset = load_dataset("lmms-lab/POPE", cache_dir=cache_dir)
    # The dataset typically holds the 'test' split.
    return pope_dataset['test']

def compute_pope_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Computes standard POPE metrics: Accuracy, Precision, Recall, F1, Yes Ratio.
    """
    assert len(predictions) == len(labels), "Predictions and labels length mismatch."
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for pred_text, label in zip(predictions, labels):
        ans = str(pred_text).lower()
        label = str(label).lower()
        
        # Tokenize simply to look for yes/no words standalone
        words = set(re.findall(r'\b\w+\b', ans))
        
        # Logic from standard POPE evaluation
        if 'yes' in words and 'no' not in words:
            pred = 'yes'
        elif 'no' in words and 'yes' not in words:
            pred = 'no'
        elif 'yes' in words and 'no' in words:
            # Fallback to the first occurrence
            pred = 'yes' if ans.index('yes') < ans.index('no') else 'no'
        else:
            # Depending on strictness, usually fallback to 'no' or considered wrong.
            pred = 'no'
            
        if pred == 'yes' and label == 'yes':
            TP += 1
        elif pred == 'yes' and label == 'no':
            FP += 1
        elif pred == 'no' and label == 'yes':
            FN += 1
        elif pred == 'no' and label == 'no':
            TN += 1
            
    total = len(labels)
    acc = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = (TP + FP) / total if total > 0 else 0

    return {
        "Total Count": total,
        "Accuracy": acc * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100,
        "Yes Ratio": yes_ratio * 100
    }

class POPEEvaluator:
    def __init__(self, cache_dir="~/POPE"):
        self.dataset = load_pope_dataset(cache_dir)
        
    def evaluate(self, predictions_dict: Dict[Union[int, str], str]):
        """
        predictions_dict: Dictionary mapping question_id to model's string response.
        Returns evaluation metrics dictionary.
        """
        preds = []
        labels = []
        
        for item in self.dataset:
            # Retrieve the identifier depending on exact schema of POPE dataset
            # (Sometimes it's 'question_id', sometimes just 'id')
            q_id = item.get('question_id', item.get('id', None))
            # Also it's useful to cast purely to string just in case keys from JSON are strings
            str_q_id = str(q_id)
            
            # Predict matching
            found_ans = predictions_dict.get(q_id, predictions_dict.get(str_q_id, None))
            
            if found_ans is not None:
                preds.append(found_ans)
                labels.append(item['label'])
                
        if not preds:
            logger.warning("No matching IDs found between predictions and POPE dataset.")
            return {}
            
        return compute_pope_metrics(preds, labels)

def read_predictions(file_path: str) -> Dict[str, str]:
    predictions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    q_id = item.get('question_id', item.get('id', None))
                    ans = item.get('text', item.get('answer', str(item)))
                    if q_id is not None:
                        predictions[str(q_id)] = ans
            elif isinstance(data, dict):
                predictions = {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            # Try reading as JSONL
            f.seek(0)
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q_id = item.get('question_id', item.get('id', None))
                ans = item.get('text', item.get('answer', str(item)))
                if q_id is not None:
                    predictions[str(q_id)] = ans
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate standard POPE metrics")
    parser.add_argument("--input_file", type=str, required=True, help="Path to predictions JSON/JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--cache_dir", type=str, default="~/POPE", help="Cache dir for POPE HF dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    print(f"Reading predictions from {args.input_file}...")
    predictions = read_predictions(args.input_file)
    print(f"Loaded {len(predictions)} prediction entries.")

    print(f"Initializing POPE evaluator...")
    evaluator = POPEEvaluator(cache_dir=args.cache_dir)
    
    print("Evaluating metrics...")
    metrics = evaluator.evaluate(predictions)
    
    if metrics:
        # Write to output file
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Successfully evaluated! Results saved to {args.output_file}")
        print(json.dumps(metrics, indent=4))
    else:
        print("Evaluation failed. Could not compute metrics (possibly no matching IDs).")
