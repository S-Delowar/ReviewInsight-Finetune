# src/evaluation/compute_metrics.py
import json
import evaluate
from src.training.model_loader import load_tokenizer
import numpy as np
import re

# Load BERTScore
bertscore = evaluate.load("bertscore")
tokenizer = load_tokenizer()

def compute_semantic_f1(preds, refs):
    """Compute average semantic F1 score using BERTScore."""
    if not preds or not refs:
        return 0.0
    results = bertscore(predictions=preds, references=refs, lang="en")
    return float(sum(results["f1"]) / len(results["f1"]))


def extract_json(text: str) -> dict | None:
  """
  Try to safely extract JSON from text.
  Returns dict if valid, else None.
  """
  if not text:
      return None
  # Remove leading/trailing whitespace
  text = text.strip()

  # Try direct load first
  try:
      return json.loads(text)
  except json.JSONDecodeError:
      pass

  # Try extracting JSON using regex
  match = re.search(r"\{.*\}", text, re.DOTALL)
  if match:
      try:
          return json.loads(match.group())
      except json.JSONDecodeError:
          # As a last resort, truncate to last closing brace
          truncated = match.group()
          last_brace = truncated.rfind("}")
          if last_brace != -1:
              try:
                  return json.loads(truncated[:last_brace+1])
              except json.JSONDecodeError:
                  return None
  return None


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
        
    # JSON validity
    valid_json_count = 0
    pros_preds_strict, cons_preds_strict, pros_labels_strict, cons_labels_strict = [], [], [], []
    pros_preds_loose, cons_preds_loose, pros_labels_loose, cons_labels_loose = [], [], [], []

    for pred, label in zip(decoded_preds, decoded_labels):
        p_json = extract_json(pred)
        l_json = extract_json(label)

        # Strict: only valid JSON
        if p_json and l_json:
            valid_json_count += 1
            pros_preds_strict.append(" ".join(p_json.get("pros", [])))
            cons_preds_strict.append(" ".join(p_json.get("cons", [])))
            pros_labels_strict.append(" ".join(l_json.get("pros", [])))
            cons_labels_strict.append(" ".join(l_json.get("cons", [])))

        # Loose: fallback to raw text if JSON invalid
        if p_json:
            pros_preds_loose.append(" ".join(p_json.get("pros", [])))
            cons_preds_loose.append(" ".join(p_json.get("cons", [])))
        else:
            pros_preds_loose.append(pred)
            cons_preds_loose.append(pred)

        if l_json:
            pros_labels_loose.append(" ".join(l_json.get("pros", [])))
            cons_labels_loose.append(" ".join(l_json.get("cons", [])))
        else:
            pros_labels_loose.append(label)
            cons_labels_loose.append(label)

    # Metrics
    json_validity_rate = valid_json_count / max(1, len(decoded_preds))

    return {
        # Format quality
        "json_validity_rate": json_validity_rate,
        "valid_json_count": valid_json_count,

        # Strict semantic quality (valid JSON only)
        "pros_semantic_f1_strict": compute_semantic_f1(pros_preds_strict, pros_labels_strict),
        "cons_semantic_f1_strict": compute_semantic_f1(cons_preds_strict, cons_labels_strict),

        # Loose semantic quality (fallback allowed)
        "pros_semantic_f1_loose": compute_semantic_f1(pros_preds_loose, pros_labels_loose),
        "cons_semantic_f1_loose": compute_semantic_f1(cons_preds_loose, cons_labels_loose),
    }