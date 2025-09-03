import json
import torch

from src.inference.loader import load_model

tokenizer, model = load_model()

def reviews_to_bullets(reviews):
    return "\n".join([f"- {r.strip()}" for r in reviews if isinstance(r, str) and r.strip()])

def build_messages(reviews: list[str]):
    instruction = "Extract pros and cons from the following reviews."
    user_content = f"Instruction: {instruction}\nReviews:\n{reviews_to_bullets(reviews)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


SYSTEM_PROMPT = "You are an assistant that extracts pros and cons from product reviews. Return only valid JSON with keys pros and cons."

def generate_pros_cons(reviews: list[str], max_new_tokens=512):
    messages = build_messages(reviews)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract JSON only
    try:
        json_start = decoded.index("{")
        json_end = decoded.rindex("}") + 1
        return json.loads(decoded[json_start:json_end])
    except Exception:
        return {"pros": [], "cons": [], "raw_output": decoded}
