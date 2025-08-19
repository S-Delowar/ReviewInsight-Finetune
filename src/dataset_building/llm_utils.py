import json
from openai import OpenAI
from dotenv import load_dotenv
import os

from src.utils.config_loader import load_config

load_dotenv() 

config = load_config()

model_name = config["generation"]["model"]
temperature = config["generation"]["temperature"]

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


SYSTEM_PROMPT = """You are an assistant that extracts **3-5 pros and 3-5 cons** from product reviews.

### Instructions:
- Read the given list of reviews about a product.
- Summarize them into clear **pros** and **cons**.
- Each pro/con must be **short (max 7 words)**.
- Aim for ** 1 to 5 pros** and **1 to 5 cons**, depending on the reviews.
- Do NOT include duplicates or overly specific details.
- Keep answers **concise, structured, and helpful**.
- Always return valid **JSON only** with two keys: "pros" and "cons".

### Example Input Reviews:
[
  "The picture quality is excellent, very sharp and clear.",
  "The cable feels loose and sometimes disconnects easily.",
  "Setup was super easy, worked immediately with my TV.",
  "It’s a bit pricey compared to other options.",
  "The device requires a separate power adapter to function.",
  "Perfect for travel, I used it in a hotel room.",
  "Not compatible with my older HD+ Nook device."
]

### Example Output:
{
  "pros": [
    "Excellent picture quality",
    "Easy to set up",
    "Great for travel use"
  ],
  "cons": [
    "Loose cable connection",
    "Price is high",
    "Requires separate power adapter",
    "Not compatible with older devices"
  ]
}
"""

def generate_pros_cons(reviews: list[str]) -> dict:
    """
    Generate pros and cons from reviews using GPT-4o-mini.
    """
    user_prompt = f"Generate pros and cons from the following reviews:\n{json.dumps(reviews, indent=2)}"
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature
    )
    
    answer_text = response.choices[0].message.content
    
    try:
        return json.loads(answer_text)
    except json.JSONDecodeError:
        return {"pros": [], "cons": [], "raw_output": answer_text}
