from openrlhf.datasets import RewardDataset
from openrlhf.utils import get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression

from transformers import AutoTokenizer
import torch 
import torch.nn.functional as F

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

model_dir = "/workspace/jihaozhe/models/UltraRM-13b"

device="cuda:0"

model = get_llm_for_sequence_regression(
        model_dir,
        "reward",
        bf16=False,
        value_head_prefix="regression_head",
        device_map=device
    )

tokenizer = get_tokenizer(model_dir, model, "left", None, use_fast=True)

# from https://huggingface.co/openbmb/UltraRM-13b
dataset = [ # cases in webgpt
  {"chosen":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Sean Connery's character, fictional British Secret Service agent James Bond, in the movie Goldfinger, stated that he preferred his martini to be \"shaken, not stirred\". [1] Some believe that Bond ordered his martini shaken because of the vodka it contained, as vodka was, for the most part, refined from potatoes (cheaper brands) which made the vodka oily. To disperse the oil, Bond ordered his martinis shaken. [2]","rejected":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Fleming's fictional British Secret Service agent James Bond orders his martini cocktail shaken, not stirred [1]. Bond's preferences for his martini are carried over to the films, where his orders are seen in both the 1961 film Dr. No and the 2006 film Casino Royale [1, 2]. In both films, Bond's subordinates copy his order, telling the bartender to keep the fruit with their drinks [2]. However, in the 2006 film, Bond appears irritated when the bartender asks if he would like his drink shaken or stirred [2]."},
  {"chosen":"Human: Sural relates to which part of the body?\nAssistant: The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].","rejected":"Human: Sural relates to which part of the body?\nAssistant: The Sural nerve runs down the side of the leg near the small saphenous vein, then passes forward below the lateral malleolus and continues on the outside of the foot as the lateral dorsal cutaneous nerve, which then communicates with the intermediate dorsal cutaneous nerve, which branches off to the side of the foot. [1]"},
]

#dataset.append({"chosen": open("ultrarm.txt", "r").read()})


chosen_input_ids = tokenizer([e["chosen"] for e in dataset],  padding=True, truncation=True, return_tensors="pt")
rejected_input_ids = tokenizer([e["rejected"] for e in dataset], padding=True, truncation=True,  return_tensors="pt")


chosen_reward = model(chosen_input_ids["input_ids"].to(device), chosen_input_ids["attention_mask"].to(device))
rejected_reward = model(rejected_input_ids["input_ids"].to(device), rejected_input_ids["attention_mask"].to(device)) 
print(chosen_reward)
print(rejected_reward)
print(chosen_reward - rejected_reward)