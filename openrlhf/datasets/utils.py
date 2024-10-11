import torch
import torch.distributed as dist
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

def zero_pad_sequences_both_sides(sequences, max_prompt_len, prompt_ids_lens, value=0):
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq, prompt_ids_len in zip(sequences, prompt_ids_lens):
        left_pad_len = max_prompt_len - prompt_ids_len
        right_pad_len = max_len - seq.size(-1) - left_pad_len
        padding = (left_pad_len, right_pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)



def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def match_input_template(text, input_template, eos_token):
    user_prompt, assist_prompt = input_template.split("{}")
    _, ua = text.split(user_prompt)
    #print(ua.split(assist_prompt))
    u, a = ua.split(assist_prompt)
    u = u.replace(eos_token, "").strip()
    a = a.replace(eos_token, "").strip()
    return u, a


def alter_template_texts(texts, old_template, new_template, old_tokenizer, new_tokenizer=None, new_max_len=1024, skip_bad_case=False):
    # old_template (input_template) format:
    # <user_prompt> {} <assistant_prompt>
    # new_template (oracle_template) format: 
    # <user_prompt> {} <assistant_prompt> {} <ending_tokens>
    new_texts = []
    for text in texts:
        if skip_bad_case:
            try:
                user_text, assist_text = match_input_template(text, old_template, old_tokenizer.eos_token)
            except:
                # use the result of previous example
                # keep the same batch size
                pass
        else:
            user_text, assist_text = match_input_template(text, old_template, old_tokenizer.eos_token)
        text = new_template.format(user_text, assist_text)
        #print(text)
        new_texts.append(text)

    if new_tokenizer is not None:
        inputs = new_tokenizer(new_texts,
                    max_length=new_max_len,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=True)

        return inputs["input_ids"], inputs["attention_mask"]
    else:
        return new_texts

if __name__ == "__main__":
    sequences = [[  523, 28766,  1838, 28766, 28767,    13,  3195,   460,   741,  7071,
         2557,  1033,   354,   272,  1707,   345,  1105,  1228,  3982, 27257,
            2, 28705,    13, 28789, 28766,   489, 11143, 28766, 28767,    13,
        28759,   294,   723, 28725, 26547,   346, 28725,   382,  2351,   525,
        28725,   662, 13860, 28725, 21950,  1007, 28725,   420,  3816,   607,
        28725, 24111, 28725,   662, 13860, 28725,   413,  1946,   440,     2,
        28705,    13, 28705,     2],
                [    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
            2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
            2,     2,     2,     2,   523, 28766,  1838, 28766, 28767,    13,
        16230, 28725,   910,   460,   368, 28804,     2, 28705,    13, 28789,
        28766,   489, 11143, 28766, 28767,    13, 28737, 28742, 28719,  2548,
         1598, 28723,  1602,   541,   315,  1316,   368,  3154, 28804,     2,
        28705,    13, 28705,     2]]

    from transformers import AutoTokenizer
    old_tokenizer = AutoTokenizer.from_pretrained("/workspace/jihaozhe/models/zephyr-7b-sft-full")
    new_tokenizer = AutoTokenizer.from_pretrained("/workspace/jihaozhe/models/ArmoRM-Llama3-8B-v0.1")
    old_template = open("/workspace/jihaozhe/rm_hacking/src/template/zephyr.in", "r").read()
    print(old_template)
    new_template = open("/workspace/jihaozhe/rm_hacking/src/template/ArmoRM.in", "r").read() + open("/workspace/jihaozhe/rm_hacking/src/template/ArmoRM.out", "r").read()
    print(new_template)

    print(old_tokenizer.chat_template)

    new_seqs, new_mask = alter_template_sequences(sequences, old_template, new_template, old_tokenizer, new_tokenizer)
    print(new_seqs)
    print(new_tokenizer.decode(new_seqs[0]))
    print(new_tokenizer.decode(new_seqs[1]))



    




    