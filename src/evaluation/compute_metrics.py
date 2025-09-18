# src/evaluation/compute_metrics.py
import json
import evaluate
from src.training.model_loader import load_tokenizer
import numpy as np


# Load BERTScore
bertscore = evaluate.load("bertscore")
tokenizer = load_tokenizer()

def compute_semantic_f1(preds, refs):
    """Compute average semantic F1 score using BERTScore."""
    if not preds or not refs:
        return 0.0
    results = bertscore(predictions=preds, references=refs, lang="en")
    return float(sum(results["f1"]) / len(results["f1"]))

def is_valid_json(s: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def compute_metrics(eval_preds):
    """
    Metrics during training:
    - JSON validity rate
    - Semantic F1 for pros
    - Semantic F1 for cons
    """
    predictions, labels = eval_preds

    # Get token IDs from logits
    if isinstance(predictions, tuple): 
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1)

    # Decode predictions & labels
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Replace -100 in labels before decoding
    labels = np.array(labels)
    labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # JSON validity
    json_validity_rate = sum(is_valid_json(p) for p in decoded_preds) / max(1, len(decoded_preds))

    # Collect pros/cons text
    pros_preds, cons_preds, pros_labels, cons_labels = [], [], [], []

    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            p_json, l_json = json.loads(pred), json.loads(label)
            pros_preds.append(" ".join(p_json.get("pros", [])))
            cons_preds.append(" ".join(p_json.get("cons", [])))
            pros_labels.append(" ".join(l_json.get("pros", [])))
            cons_labels.append(" ".join(l_json.get("cons", [])))
        except Exception:
            continue

    return {
        "json_validity_rate": json_validity_rate,
        "pros_semantic_f1": compute_semantic_f1(pros_preds, pros_labels),
        "cons_semantic_f1": compute_semantic_f1(cons_preds, cons_labels),
    }
