from openrlhf.datasets import RewardDataset
from openrlhf.datasets.utils import zero_pad_sequences
from openrlhf.utils import get_tokenizer, get_strategy
from openrlhf.models import get_llm_for_sequence_regression

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse

def make_inputs(tokenizer, model_name, max_length=100, openrlhf=True, use_chat_template=True):
    prompt = 'What are some synonyms for the word "beautiful"?'
    response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
    messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]

    prompt2 = 'Hello, how are you?'
    response2 = "I'm doing great. How can I help you today?"
    messages2 = [{"role": "user", "content": prompt2},
           {"role": "assistant", "content": response2}]
    
    def make_input_openrlhf(m):
        manual_template = {"armorm": ["<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                                      "{}<|eot_id|>"],
                           "mistral": ["<s>[INST] {}[/INST] ",
                                       "{}</s>"],
                           "zephyr": ["<|user|>\n{}</s>\n<|assistant|>\n",
                                      "{}</s>"]}
        if use_chat_template:
            m = tokenizer.apply_chat_template(m, tokenize=False)
        else:
            m = manual_template[model_name][0].format(m[0]["content"]) + \
                manual_template[model_name][1].format(m[1]["content"])
        if not m.endswith(tokenizer.eos_token):
            m += " " + tokenizer.eos_token
        m = tokenizer(m, 
                max_length=max_length, 
                padding=False, 
                truncation=True, 
                return_tensors="pt", 
                add_special_tokens=False)
        m["input_ids"][0][-1] = tokenizer.eos_token_id
        m["attention_mask"][0][-1] = True 
        return m["input_ids"], m["attention_mask"]        
        
        

    if openrlhf:
        mis, mas = [], []
        for m in [messages, messages2]:
            mi, ma = make_input_openrlhf(m)
            mis.append(mi)
            mas.append(ma)

        mis = zero_pad_sequences(mis, "left", value=tokenizer.pad_token_id).squeeze(1)
        mas = zero_pad_sequences(mas, "left", value=tokenizer.pad_token_id).squeeze(1)

    else:
        res = tokenizer([messages, messages2], 
                max_length=max_length, 
                padding=True,
                return_tensors="pt",
                add_special_tokens=True)
        mis = res["input_ids"]
        mas = res["attention_mask"]

    return mis, mas 

def main():
    model_dir_map = {"armorm": "/workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1",
                     "mistral": "/workspace/jihaozhe/models/Mistral-7B-Instruct-v0.3",
                     "zephyr": "/workspace/jihaozhe/models/zephyr-7b-sft-full",
                     "pythia-1.4b": "/workspace/jihaozhe/models/pythia-1.4b-sft-full",
                     "gemma": "/workspace/jihaozhe/models/gemma-2b-zephyr-sft",
                     "llama": "/workspace/jihaozhe/models/Llama-3.2-1B-sft-full"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    


    args = parser.parse_args()

    pretrain = model_dir_map[args.model.lower()]


    cls_class = None
    if args.model.lower() == "armorm":
        cls_class = AutoModelForSequenceClassification
    
    model = get_llm_for_sequence_regression(
        pretrain,
        "reward",
        cls_class=cls_class,
        bf16=True,
    )

    tokenizer = get_tokenizer(pretrain, model, "left", None, use_fast=True)

    input_ids, attention_mask = make_inputs(tokenizer, 
                                            args.model.lower(), 
                                            100, 
                                            openrlhf=True, 
                                            use_chat_template=True)


    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    print(input_ids[0])
    print("="*50)
    print(tokenizer.decode(input_ids[1], skip_special_tokens=False))
    print(input_ids[1])

    print(attention_mask[0])
    print(attention_mask[1])

    
if __name__ == "__main__":
    main()


