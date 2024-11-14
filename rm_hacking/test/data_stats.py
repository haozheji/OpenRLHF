from datasets import load_dataset
import sys 
from transformers import AutoTokenizer
import numpy as np 
import tqdm

data_dir = sys.argv[-2]
model_dir = sys.argv[-1]

tokenizer = AutoTokenizer.from_pretrained(model_dir)

dataset = load_dataset(data_dir)["train"]

'''
for k in dataset[0].keys():
    if "chosen" in k:
        ckey = k 
    elif "reject" in k:
        rkey = k 
    elif "instruction" in k:
        pkey = k
'''
ckey = "chosen_response"
rkey = "rejected_response"
pkey = "instruction"

print(ckey, rkey, pkey)
prompts, ctoks, rtoks = [], [], []
for line in tqdm.tqdm(dataset):
    prompts.append(len(tokenizer.encode(line[pkey])))
    #print(line[ckey])
    ctoks.append(len(tokenizer.encode(line[ckey])))
    rtoks.append(len(tokenizer.encode(line[rkey])))

#prompts = np.array(prompts)
#ctoks = np.array(ctoks)
#rtoks = np.array(rtoks)

_mean = lambda x: sum(x) / len(x)
_min = lambda x: min(x)
_max = lambda x: max(x)

meanlen = {"prompt": _mean(prompts),
           "chosen": _mean(ctoks),
           "rejected": _mean(rtoks)}

maxlen = {"prompt": _max(prompts),
           "chosen": _max(ctoks),
           "rejected": _max(rtoks)}

minlen = {"prompt": _min(prompts),
           "chosen": _min(ctoks),
           "rejected": _min(rtoks)}

print("mean length:")
for k, v in meanlen.items():
    print(f"{k}: {v}")
    
print("max length:")
for k, v in maxlen.items():
    print(f"{k}: {v}")

print("min length:")
for k, v in minlen.items():
    print(f"{k}: {v}")

