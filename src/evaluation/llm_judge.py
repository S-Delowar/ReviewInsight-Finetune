import random
import wandb
import json
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv


from src.utils.config_loader import load_config

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

llm_judge_model = load_config()["generation"]["model"]

def run_llm_judge(model, tokenizer, test_data, device, subset_size=100):
    """
    Run qualitative evaluation using an LLM as a judge.
    - Samples a subset from test data
    - Generates predictions with the fine-tuned model
    - Compares predictions to references
    - Uses an external LLM (judge) to score quality
    - Returns a wandb.Table with structured results
    """
    qual_samples = random.sample(list(test_data), min(subset_size, len(test_data)))
    
    table = wandb.Table(
        columns=[
            "instruction",
            "model_response",
            "reference",
            "json_validity_judge",
            "pros_score",
            "cons_score",
            "judge_comment"
        ]
    )
    
    # Ensure proper device handling
    device = torch.device(device)
    
    for sample in qual_samples:
        # For wandb logging only
        instruction_text = f"{sample['instruction']}\nReviews:\n" + "\n".join([f"- {review}" for review in sample["input"]])
        
        
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False 
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=False
            )
            
        # Decode
        model_response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # Reference
        ref_text = sample["answer"]
            
        # === LLM-as-Judge: Structured Scoring ===
        judge_prompt = f"""
        You are a strict evaluator. Compare the model output with the reference.

        Reference JSON:
        {ref_text}

        Model Output JSON:
        {response}

        Evaluate on:
        - JSON validity (1 if valid, 0 if invalid) — use your own parsing, don't trust the model's claim
        - Pros correctness: How accurately are pros described? (0-5 scale, 5 = fully correct and complete)
        - Cons correctness: How accurately are cons described? (0-5 scale, 5 = fully correct and complete)

        Return your answer as valid JSON only:
        {{
        "json_validity": 0 or 1,
        "pros_score": 0 to 5,
        "cons_score": 0 to 5,
        "comment": "Brief rationale for scores"
        }}
        """    
        try:
            response = client.chat.completions.create(
                model=llm_judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
            )
            judge_json = response.choices[0].message.content.strip()

            # Try parsing judge output as JSON
            try:
                judge_data = json.loads(judge_json)
                pros_score = judge_data.get("pros_score", -1)
                cons_score = judge_data.get("cons_score", -1)
                json_valid_judge = judge_data.get("json_validity", -1)
                comment = judge_data.get("comment", "No comment.")
            except json.JSONDecodeError:
                pros_score = -1
                cons_score = -1
                json_valid_judge = -1
                comment = f"Judge failed to return valid JSON. Raw output: {judge_json}"

        except Exception as e:
            pros_score = -1
            cons_score = -1
            json_valid_judge = -1
            comment = f"Judge API error: {str(e)}"

        # Add row to wandb table
        table.add_data(
            instruction_text,
            model_response,
            ref_text,
            json_valid_judge,
            pros_score,
            cons_score,
            comment
        )

    return table
