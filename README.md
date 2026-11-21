# BERT Multi-Class News Classification

Fine-tuning BERT for news classification using PyTorch and Hugging Face.

## Dataset
AG News: World, Sports, Business, Sci/Tech.

## Model
- bert-base-uncased
- Training samples: 8,000
- Validation samples: 1,000
- Test samples: 1,000
- Epochs: 2
- GPU: NVIDIA Tesla T4

## Results
- Validation Accuracy: 93.1%
- Test Accuracy: 91.4%
- Training Time: 187 seconds

## Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

## Example Prediction
Input: NASA announces a new mission to explore Mars.

Prediction: Sci/Tech

Softmax confidence: 97.49%

## Technologies
Python, PyTorch, Transformers, Hugging Face,
scikit-learn, pandas, Matplotlib, Google Colab.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classifier:

```bash
python predict.py
```

The fine-tuned BERT model is automatically downloaded from [Hugging Face](https://huggingface.co/uniofedin0304/bert-ag-news-classification).

## Example Prediction

**Input:** NASA announces a new mission to explore Mars.

**Predicted category:** Sci/Tech

**Confidence:** 98.08%

## Reproducibility

The fine-tuned BERT model is publicly available on [Hugging Face](https://huggingface.co/uniofedin0304/bert-ag-news-classification).

The inference pipeline was successfully tested in a fresh Google Colab environment without retraining or requiring local model files.

**Example output:**
```text
Input: NASA announces a new mission to explore Mars.
Prediction: Sci/Tech
Confidence: 98.08%
```

The model automatically downloads its weights and tokenizer from Hugging Face when running `predict.py`.
