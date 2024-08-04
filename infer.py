from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

args = dict(
  model_name_or_path="THUDM/glm-4-9b", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="saves/GLM-4-9B/lora/train_multilabel",            # load the saved LoRA adapters
  template="default",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
)
chat_model = ChatModel(args)

messages = []
print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
while True:
  query = input("\nUser: ")
  if query.strip() == "exit":
    break
  if query.strip() == "clear":
    messages = []
    torch_gc()
    print("History has been removed.")
    continue

  messages.append({"role": "user", "content": query})
  print("Assistant: ", end="", flush=True)

  response = ""
  for new_text in chat_model.stream_chat(messages):
    print(new_text, end="", flush=True)
    response += new_text
  print()
  messages.append({"role": "assistant", "content": response})

torch_gc()