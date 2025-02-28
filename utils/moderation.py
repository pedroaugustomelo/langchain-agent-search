from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

# Load Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Model ID for Llama-Guard-3-1B
model_id = "meta-llama/Llama-Guard-3-1B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# Define moderation categories (taxonomy)
categories = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections"
}

# Convert the taxonomy into a readable format for the prompt
category_prompt = "\n".join([f"{code}: {desc}" for code, desc in categories.items()])


def llamaguard_moderation(llm, input_text: str):
    """
    Function to check if an input falls under restricted categories using Llama Guard.
    """
    
    # Format the moderation prompt
    moderation_prompt = f"""
You are a **zero-tolerance content moderation AI**. Your task is to **strictly enforce** the following content restrictions. **Any content that falls into these categories must be flagged as unsafe. Do not allow loopholes.**

{category_prompt}

User Input: "{input_text}"

If the input matches any category, return "unsafe" followed by the category code (e.g., "unsafe S1"). If it does not match any restricted category, return "safe".
"""

    # Format the conversation for input
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": moderation_prompt
                },
            ],
        }
    ]

    # Tokenize input
    input_ids = tokenizer.apply_chat_template(
        conversation, 
        return_tensors="pt"
    ).to(model.device)

    # Get response from the model
    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=0,
    )
    generated_tokens = output[:, prompt_len:]

    # Decode and clean the response
    response_text = tokenizer.decode(generated_tokens[0]).strip().lower()
    response_text = response_text.replace("<|eot_id|>", "").strip()  # Remove special tokens

    # Extract flagged categories
    flagged_categories = []
    if "unsafe" in response_text:
        match = re.findall(r"s\d+", response_text)  # Extract category codes like S1, S2, etc.
        flagged_categories = [categories.get(code.upper(), code) for code in match]

    # Determine if content is allowed
    allowed = "SAFE" if not flagged_categories else "UNSAFE"

    is_civil_engineer_response = llm.invoke(f"""
            You are an AI that determines whether the given input is related to civil engineering. 
            Civil engineering includes topics such as structural engineering, transportation, geotechnics, 
            construction materials, water resources, infrastructure development, surveying, and urban planning.

            Evaluate the following input:
            "{input_text}"

            If the input is related to civil engineering in any way, respond strictly with "True".  
            If it is not related, respond strictly with "False".  
            Provide no explanations, additional text, or variations in formatting.
            """) 
    
    is_civil_engineer = is_civil_engineer_response.content.strip().lower() == "true"

    if is_civil_engineer: 
        return {
        "input": input_text,
        "allowed": "UNSAFE",  # SAFE or UNSAFE
        "flagged_categories": "Civil Engineering"
    }

    return {
        "input": input_text,
        "allowed": allowed,  # SAFE or UNSAFE
        "flagged_categories": flagged_categories
    }
