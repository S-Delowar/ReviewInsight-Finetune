from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.inference import generate_pros_cons

app = FastAPI(title="Review Insight API")

class InferenceRequest(BaseModel):
    instruction: str
    reviews: List[str]

class InferenceResponse(BaseModel):
    pros: List[str]
    cons: List[str]
    raw_output: str | None = None


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    result = generate_pros_cons(request.instruction, request.reviews)
    return InferenceResponse(**result)
