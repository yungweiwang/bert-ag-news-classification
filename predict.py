
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "uniofedin0304/bert-ag-news-classification"

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()

    return {
        "category": LABELS[predicted_class],
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    text = input("Enter a news headline: ")
    result = predict(text)
    print(result)


