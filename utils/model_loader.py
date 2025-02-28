from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Detect device (For Oracle Linux, force CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Loads the Llama Guard model and tokenizer into memory once, optimized for CPU."""
    global model, tokenizer
    try:
        print(f"🔥 Loading Llama Guard model on {device.upper()}...")

        # ✅ If GPU is available, use it with bfloat16
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda")

        # ✅ If CPU, use standard model (NO bitsandbytes)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # ✅ Use float32 for CPU
                device_map={"": "cpu"},  # ✅ Explicitly force CPU usage
            )

        # ✅ Enable torch.compile() for PyTorch 2.0+ (Speeds up inference)
        if torch.__version__ >= "2.0":
            model = torch.compile(model)

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        print(f"✅ Model successfully loaded on {device.upper()}!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise Exception("Model loading failed")

# Load the model before Flask starts
load_model()

# Export model and tokenizer
__all__ = ["model", "tokenizer", "device"]
