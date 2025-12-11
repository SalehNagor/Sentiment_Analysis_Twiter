from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import logging

from src.logging_utils import setup_logging

# Configure logging for the API
setup_logging()
logger = logging.getLogger(__name__)


class SentimentRequest(BaseModel):
    text: str


app = FastAPI(title="Sentiment Analysis API")

MODEL_PATH = "./models/distilbert_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ID to label mapping (must match training)
ID2LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    logger.info("Model and tokenizer loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logger.error("Error loading model from %s: %s", MODEL_PATH, e)
    print(f"Error loading model: {e}")
    print("Please make sure you ran 'python main.py' first to train the model.")


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Use /predict to classify text."}


@app.post("/predict")
def predict(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
        predicted_class_id = int(torch.argmax(probs).item())

    label = ID2LABEL.get(predicted_class_id, "Unknown")
    confidence = float(probs[predicted_class_id].item())

    logger.info(
        "Prediction - text: %s | label: %s | confidence: %.4f",
        request.text,
        label,
        confidence,
    )

    return {
        "text": request.text,
        "sentiment": label,
        "confidence": f"{confidence:.4f}",
        "probabilities": {
            "Negative": f"{probs[0].item():.4f}",
            "Neutral": f"{probs[1].item():.4f}",
            "Positive": f"{probs[2].item():.4f}",
        },
    }