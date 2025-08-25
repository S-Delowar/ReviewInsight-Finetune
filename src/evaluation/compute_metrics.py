# src/evaluation/compute_metrics.py
import evaluate
import json
from src.training.model_loader import load_tokenizer



# Load Hugging Face BERTScore via evaluate
bertscore = evaluate.load("bertscore") 

def compute_semantic_f1(preds, refs) -> float:
    """
    Compute semantic F1 using BERTScore.
    """
    if len(preds) == 0 and len(refs) == 0:
        return 1.0
    if len(preds) == 0 or len(refs) == 0:
        return 0.0

    results = bertscore(predictions=preds, references=refs, lang="en")
    f1_values = results["f1"]
    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0



# Load tokenizer
tokenizer = load_tokenizer()

def is_valid_json(pred_str: str) -> bool:
    try:
        json.loads(pred_str)
        return True
    except Exception:
        return False

def compute_metrics(eval_preds):
    """
    Compute metrics (JSON validity rate, Pros/Cons F1 scores) during training/evaluation.
    """
    predictions, labels = eval_preds

    # Decode predictions and labels
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]

    # JSON validity
    valid_jsons = sum(is_valid_json(p) for p in decoded_preds)
    json_validity_rate = valid_jsons / len(decoded_preds) if decoded_preds else 0.0

    # Compute Pros F1 / Cons F1 (Semantic) if both available in label JSONs
    pros_preds, cons_preds, pros_labels, cons_labels = [], [], [], []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            pred_json = json.loads(pred)
            label_json = json.loads(label)
            pros_preds.append(" ".join(pred_json.get("pros", [])))
            cons_preds.append(" ".join(pred_json.get("cons", [])))
            pros_labels.append(" ".join(label_json.get("pros", [])))
            cons_labels.append(" ".join(label_json.get("cons", [])))
        except Exception:
            continue
    
    pros_semantic_f1 = compute_semantic_f1(pros_preds, pros_labels)
    cons_semantic_f1 = compute_semantic_f1(cons_preds, cons_labels)
    
    return {
        "json_validity_rate": json_validity_rate,
        "pros_semantic_f1": pros_semantic_f1,
        "cons_semantic_f1": cons_semantic_f1,
    }