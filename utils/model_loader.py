# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Set device (Prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        print("🔥 Loading Llama Guard model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="cuda" if device == "cuda" else "cpu",
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise Exception("Model loading failed")

# Load model synchronously before Flask app starts
load_model()

__all__ = ["model", "tokenizer", "device"]
