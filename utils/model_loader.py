# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import threading

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Set device (Prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = False

def load_model():
    global model, tokenizer, model_loaded
    try:
        print("🔥 Loading Llama Guard model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="cuda" if device == "cuda" else "cpu",
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model_loaded = True
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Start the model loading process in the background thread
def start_model_loading():
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()

# Call this function when the app starts to begin loading the model asynchronously
start_model_loading()

__all__ = ["model", "tokenizer", "device", "model_loaded"]
