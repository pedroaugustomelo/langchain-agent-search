from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Set device (Prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model ONCE
print("🔥 Loading Llama Guard model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="cuda" if device == "cuda" else "cpu",
).to(device)

# Load tokenizer ONCE
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

print("✅ Model loaded successfully!")

# Export model and tokenizer so other scripts can import them
__all__ = ["model", "tokenizer", "device"]
