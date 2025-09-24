# Product Review Insights with Mistral-7B-Instruct-v0.2 (LoRA Fine-Tuned)
**"Automatically extract structured pros and cons from product reviews using a lightweight, instruction-tuned LLM."** 

This project provides a fine-tuned version of **Mistral-7B-Instruct-v0.2** using **LoRA (Low-Rank Adaptation)** for the specific task of extracting structured pros and cons from customer product reviews. The model outputs clean, standardized JSON — perfect for e-commerce platforms, market research, or automated feedback analysis.

##  Features
- ✅ Fine-tuned **Mistral-7B-Instruct-v0.2** with **LoRA adapters** (parameter-efficient).
- ✅ Trained on **10,000 curated Amazon product reviews** → distilled into structured pros/cons.
- ✅ Output: Valid **JSON** with `pros` and `cons` keys.
- ✅ High JSON validity rate: **98.7%**.
- ✅ Semantic F1 (Pros/Cons): **0.89 / 0.87**.
- ✅ Qualitative evaluattion with **LLM-as-Judge (GPT-4o-mini)**.
- ✅ **W&B integration** for experiment tracking and evaluation.
- ✅ Deployed as REST API using FastAPI + Modal (GPU-accelerated).
- ✅ Quantized inference (4-bit NF4) for low memory usage.

---

## Dataset Overview
We have build custom dataset for this project.

You can find details on Huggingface Repo:
[sdelowar2/product_reviews_insight_10k](https://huggingface.co/datasets/sdelowar2/product_reviews_insight_10k)

- **Source**: Amazon product reviews, curated for **structured pros/cons extraction**.
- **Size**: 10,000 samples (training set).
- **Format**: Parquet with fields: `instruction`, `input` (list of reviews), `answer` (JSON with `pros`/`cons`).
- **Preprocessing**: Cleaned, filtered, summarized, and distilled using GPT-4o-mini.
- **Usage**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("sdelowar2/product_reviews_insight_10k")
  ```
- **Example record:**
```json
{
  "instruction": "Generate pros and cons from the following product reviews.",
  "input": [
    "Great sound quality but the battery drains fast.",
    "Comfortable to wear for long hours.",
    "Bluetooth disconnects sometimes.",
    "Excellent value for the price."
  ],
  "answer": {
    "pros": [
      "Great sound quality",
      "Comfortable to wear",
      "Good value for price"
    ],
    "cons": [
      "Battery drains fast",
      "Bluetooth disconnects"
    ]
  }
}
```

---

## Model Overview
**Base Model**
- **mistralai/Mistral-7B-Instruct-v0.2** — powerful 7B parameter instruction-following model

**Fine-Tuning Method**
- **PEFT + LoRA** (Parameter-Efficient Fine-Tuning)
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: q_proj, k_proj, v_proj, o_proj

**Training Details**
- Epochs: 3
- Optimizer: AdamW
- Learning Rate: 3e-4
- Batch Size: 2 (with gradient accumulation = 8)
- Precision: BF16 → FP16 fallback
- Max Sequence Length: 512

**Evaluation Metrics**
<custom-element data-json="%7B%22type%22%3A%22table-metadata%22%2C%22attributes%22%3A%7B%22title%22%3A%22Evaluation%20Metrics%22%7D%7D" />

| Metric               | Score  |
|----------------------|--------|
| JSON Validity Rate   | 98.7%  |
| Semantic F1 (Pros)  | 0.89   |
| Semantic F1 (Cons)  | 0.87   |
| BERTScore            | 0.85   |

- JSON Validity Rate: % of outputs with valid JSON structure.
- Semantic F1 (Pros/Cons): Strict and loose F1 scores for semantic accuracy.
- BERTScore: Measures semantic similarity between predictions and ground truth.

---

## Quick Start with the Fine-tuned Model
### Installation
```python
pip install -q -U bitsandbytes==0.42.0
pip install -q -U peft==0.8.2
pip install -q -U accelerate==0.27.0
pip install -q -U transformers==4.38.0
```

### Load Model & Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_id = "sdelowar2/mistral-7B-reviews-insight-lora"

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_id)
```

### Inference Example

```python
reviews = [
  "I am using this with a Nook HD+. It works as described. The HD picture on my Samsung 52",
  "The price is completely unfair and only works with the Nook HD and HD+. The cable is very wobb",
  "This adaptor is real easy to setup and use right out of the box. I had not problem with it",
  "This adapter easily connects my Nook HD 7&#34; to my HDTV through the HDMI cable.",
  "Gave it five stars because it really is nice to extend the screen and use your Nook as a streaming"
]

SYSTEM_PROMPT = "You are an assistant that extracts pros and cons from product reviews. Return only valid JSON with keys pros and cons."

user_content = "Instruction: Extract pros and cons from the following reviews.\nReviews:\n" + "\n".join(f"- {r}" for r in reviews)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    add_generation_prompt=True 
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
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

try:
    parsed = json.loads(response)
    print(f"result: {parsed}")
except Exception as e:
    print("\nCould not parse JSON:", e)
    print(f"raw output: {response}")

```

---

### REST API (FastAPI + Modal)
Deployed endpoint using fastapi and modal for scalable, low-latency inference.

**Endpoint**

Send POST request to:
[https://mdelowar2230053--review-insights-llm-fastapi-app.modal.run/generate/](https://mdelowar2230053--review-insights-llm-fastapi-app.modal.run/generate/) 

**Request Body (JSON)**
```json
{
    "reviews":[
      "This one worked immediately right out of the box and is everything it is decribed to be. Fast service from",
      "The drive works with software other that ACER's eRecovery but note that returns subtract 15% of the",
      "The Acer Aspire One USB External DVD/CDRW CD Burner suits my need perfectly. For my",
      "Good value. Works as described. A little bigger than it needs to be, but does the job.",
      "Tengo una Mini y no tiene unidad de disco, so use this producto",
      "External drive installed and seemed to do its stuff without any problems on WIN7. Comes with an extra USB cable",
      "The Acer Aspire Netbook is sold with Windows XP operating system. This DVD/CD burner is sell with",
      "This plays CD's and DVD's as well as any internal drive in a laptop. No big deal to pack",
      "I have had this for a couple months. Used it to load Win7 on a Acer Netbook. Work"
]
  }
```

**Response**
```json
{
	"result": {
		"pros": [
			"Works immediately out of the box",
			"Good value for the price",
			"Compatible with various software",
			"Plays CDs and DVDs well",
			"Fast service from the vendor"
		],
		"cons": [
			"A bit bigger than necessary",
			"Returns reduce refund value"
		]
	}
}
```

---

## Evaluation with LLM-as-Judge
We evaluated model outputs on a test set using **GPT-4o-mini** as a judge. Metrics logged to **Weights & Biases** include:

- JSON Validity (0/1)
- Pros Accuracy Score (0-5)
- Cons Accuracy Score (0-5)
- Judge’s qualitative comment

See src/evaluation/llm_judge.py for implementation.

---

## Monitoring & Logging
All training and evaluation runs are tracked via **Weights & Biases (W&B)**.

- Training loss, learning rate, GPU utilization
- Qualitative LLM-judge evaluations in interactive tables
