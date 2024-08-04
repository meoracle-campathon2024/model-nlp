from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

from flask import Flask, request, jsonify
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

import os
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

pre_mes = """1. Neurological and Psychological
2. Respiratory
3. Gastrointestinal
4. Genitourinary
5. Musculoskeletal
6. Dermatological
7. Eye, Ear, Nose, Throat
8. Cardiovascular
9. Other General Disease"""

os.environ["GOOGLE_API_KEY"] = "AIzaSyDs4_ZIRVXjQVSYqm2o0hrhR1-n0ximGxg"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def extract_digit(text):
    match = re.search(r"\d+", text)
    return match.group() if match else None

def chatgpt(message):
    #print(message)
    response = model.generate_content(
        [message],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    #print(response)
    return response.text

def classify(disease):
    chatgpt_res = chatgpt(pre_mes + '\n\n"' + disease + '" belong to which number?')
    digitt = extract_digit(chatgpt_res)
    if digitt is not None:
        return int(digitt)
    else:
        return 9
    
# Initialize the model with the given arguments
args = dict(
  model_name_or_path="Qwen/Qwen1.5-0.5B-Chat", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="saves/Qwen1.5-0.5B-Chat/lora/qwen_full_data",            # load the saved LoRA adapters
  template="qwen",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
)

# args = dict(
#   model_name_or_path="google/gemma-2b", # use bnb-4bit-quantized Llama-3-8B-Instruct model
#   adapter_name_or_path="saves/Gemma-2B/lora/Gemma_2B_fix",            # load the saved LoRA adapters
#   template="default",                     # same to the one in training
#   finetuning_type="lora",                  # same to the one in training
#   quantization_bit=4,                    # load 4-bit quantized model
# )
chat_model = ChatModel(args)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    query = data['query']
    messages = []

    messages.append({"role": "user", "content": query})
    print("Assistant: ", end="", flush=True)
    response = ""
    for new_text in chat_model.stream_chat(messages):
        print(new_text, end="", flush=True)
        response += new_text
    response.split()
    return jsonify([{'name': response, 'category_id': classify(response)}])

if __name__ == '__main__':
    app.run(port=5000)  # Change the port number if you want to use a different one
