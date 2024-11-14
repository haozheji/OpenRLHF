from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model_dir = "/workspace/jihaozhe/models/Mistral-7B-Instruct-v0.3" #UltraRM-13b
#model_dir = "/workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

print(tokenizer.decode(model_inputs[0]))
model.to(device)

#generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
#decoded = tokenizer.batch_decode(generated_ids)
#print(decoded[0])