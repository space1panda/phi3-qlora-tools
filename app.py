from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "checkpoints/phi3_mini_q4"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
    )


# Define request body

class PhiConfig(BaseModel):
    system: str
    user: str
    generation_config: Dict[str, Any]


@app.post("/predict")
async def predict(input_data: PhiConfig):
    try:
        messages = [
            {"role": "system", "content": input_data.system},
            {"role": "user", "content": input_data.user}
        ]

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        output = pipe(messages, **input_data.generation_config)
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
