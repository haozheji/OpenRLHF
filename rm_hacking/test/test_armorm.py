from openrlhf.datasets import RewardDataset
from openrlhf.utils import get_tokenizer
from openrlhf.models import get_llm_for_sequence_regression

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


model_dir = "/workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1"

device="cuda"

model = get_llm_for_sequence_regression(
        model_dir,
        "reward",
        cls_class=AutoModelForSequenceClassification,
        device_map=device,
        bf16=False
    )

tokenizer = get_tokenizer(model_dir, model, "left", None, use_fast=True)


prompt = 'What are some synonyms for the word "beautiful"?'
response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]


input_ids = tokenizer.apply_chat_template(messages, tokenize=False)

messages2 = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response+ "!"*100}]

input_ids2 = tokenizer.apply_chat_template(messages2, tokenize=False)

input_ids = tokenizer(input_ids, return_tensors="pt")["input_ids"][0]
input_ids2 = tokenizer(input_ids2, return_tensors="pt")["input_ids"][0]

amask = tokenizer(input_ids, return_tensors="pt")["attention_mask"][0]
amask2 = tokenizer(input_ids2, return_tensors="pt")["attention_mask"][0]

input_ids = zero_pad_sequences([input_ids, input_ids2], side="left", value=tokenizer.pad_token_id).squeeze(1).to(device)
amask = zero_pad_sequences([amask, amask2], side="left", value=tokenizer.pad_token_id).squeeze(1).to(device)

print(tokenizer.decode(input_ids[0]))

with torch.no_grad():
    output = model(input_ids, amask)
    gating_output = output.gating_output.cpu().float()
    preference_score = output.score.cpu().float() 
    multi_obj_rewards = output.rewards.cpu().float()  

obj_transform = model.reward_transform_matrix.data.cpu().float()
# The final coefficients assigned to each reward objective
multi_obj_coeffs = gating_output @ obj_transform.T

K = 3
top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True,)[:, :K]
top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)


attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']

example_index = 0
for i in range(K):
   attribute = attributes[top_obj_dims[example_index, i].item()]
   coeff = top_obj_coeffs[example_index, i].item()
   print(f"{attribute}: {round(coeff,5)}")

helpsteer_rewards_pred = multi_obj_rewards[0, :5] * 5 - 0.5
print(helpsteer_rewards_pred)
print(preference_score)
