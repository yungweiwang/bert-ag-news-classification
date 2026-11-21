
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_PATH = "./bert-ag-news-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model.to(device)
model.eval()

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = probabilities.argmax().item()

    return {
        "category": model.config.id2label[predicted_id],
        "confidence": round(
            probabilities[predicted_id].item(), 4
        )
    }

if __name__ == "__main__":
    text = input("Enter a news article: ")
    print(predict_news(text))
