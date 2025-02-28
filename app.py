import os
import logging
from flask import Flask, request, jsonify
from agents.graph import run_graph
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

# Load environment variables from .env file
load_dotenv()

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Set device (Prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model ONCE (kept in memory)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Use lower precision for GPU
    device_map="cuda" if device == "cuda" else "cpu",
).to(device)

# Load tokenizer ONCE (kept in memory)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Ensure request is JSON
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Expected JSON."}), 400
        
        data = request.get_json()
        
        # Validate input
        if "user_input" not in data or not isinstance(data["user_input"], str) or not data["user_input"].strip():
            return jsonify({"error": "Invalid input. 'user_input' must be a non-empty string."}), 400
        
        user_input = data["user_input"].strip()
        result = run_graph(user_input)
        return jsonify({"response": result})
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host=host, port=port, debug=True, use_reloader=False)
