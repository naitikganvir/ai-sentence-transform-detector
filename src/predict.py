# src/predict.py
import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def load_model(model_dir="naitikganvir/sentence-transformer-model", device=None, hf_token=None):
    """
    Load model directly from Hugging Face repo.
    If private repo, provide hf_token (Streamlit secrets)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Loading model from Hugging Face: {model_dir}")

    # If token is provided for private repo
    if hf_token:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_auth_token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=hf_token)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(device)

    # Load label encoder
    label_encoder_path = os.path.join("label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        le = joblib.load(label_encoder_path)
        print("✅ Label encoder loaded from local repo")
    else:
        # If not included, create dummy labels (replace with your actual labels)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(["class1", "class2"])  
        print("⚠️ Label encoder not found. Using dummy labels!")

    return model, tokenizer, le, device


def predict(sentence, model, tokenizer, label_encoder, device, return_attention=False):
    model.eval()
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=return_attention)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([pred_idx])[0]

    attn = None
    if return_attention:
        last_layer = outputs.attentions[-1][0].cpu().numpy()
        cls_to_tokens = last_layer[:, 0, :].mean(axis=0)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attn = list(zip(tokens, cls_to_tokens.tolist()))

    return {"label": label, "confidence": float(probs[pred_idx]), "probs": probs.tolist(), "attention": attn}
