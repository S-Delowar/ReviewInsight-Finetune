import torch
import json

# ---------------------------
# Inference function
# ---------------------------
def generate_pros_cons(tokenizer, model, reviews: list[str]):
    SYSTEM_PROMPT = (
        "You are an assistant that extracts pros and cons from product reviews. "
        "Return only valid JSON with keys pros and cons."
    )

    user_content = (
        "Instruction: Extract pros and cons from the following reviews.\nReviews:\n"
        + "\n".join(f"- {r}" for r in reviews)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=300,
            temperature=0.2,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    try:
        # return as dict
        return json.loads(response)  # return as dict
    except Exception:
        return {"raw_output": response}