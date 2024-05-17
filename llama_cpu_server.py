from flask import Flask, request, jsonify
from llama_cpp import Llama
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os 

# Create a Flask object
app = Flask("Llama server")
model = None

@app.route('/llama', methods=['POST'])
def generate_response():
    global model
    
    try:
        data = request.get_json()

        # Check if the required fields are present in the JSON data
        if 'system_message' in data and 'user_message' in data and 'max_tokens' in data:
            system_message = data['system_message']
            user_message = data['user_message']
            max_tokens = int(data['max_tokens'])

            # Prompt creation
            prompt = f"""<s>[INST] <<SYS>>
            {system_message}
            <</SYS>>
            {user_message} [/INST]"""
            
            # Check if model directory exists (pre-downloaded)
            if os.path.exists(os.path.join("/app", "models", "llama-2-7b-chat.Q2_K.gguf")):
                print("Loading model from pre-downloaded files...")
                tokenizer = AutoTokenizer.from_pretrained(os.path.join("/app", "models", "llama-2-7b-chat.Q2_K.gguf"))
                model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join("/app", "models", "llama-2-7b-chat.Q2_K.gguf"))
            else:
                # Download model using transformers library (requires access token)
                print("Downloading model from Hugging Face...")
                if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
                    print("Warning: Hugging Face access token not found! Model download might fail.")

            model_name = "llama-2-7b-chat.Q2_K.gguf"  # Replace with the actual model identifier from Hugging Face (if applicable)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
             
            # Run the model
            output = model(prompt, max_tokens=max_tokens, echo=True)
            
            return jsonify(output)

        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)