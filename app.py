import os
import logging
from flask import Flask, request, jsonify
from agents.graph import run_graph
from dotenv import load_dotenv
from utils.model_loader import model, tokenizer, model_loaded  # Import model and loading status

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/chat", methods=["POST"])
def chat():
    if not model_loaded:
        return jsonify({"error": "Model is still loading. Try again later."}), 503

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Expected JSON."}), 400
        
        data = request.get_json()
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
    if not model_loaded:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_RUN_PORT", 5000)), debug=True, use_reloader=False)
