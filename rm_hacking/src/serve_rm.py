import argparse
import re

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

from transformers import AutoModelForSequenceClassification


logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args, cls_class=None):
        # Modify the reward_model to your remote model
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            cls_class=cls_class,
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        print("reward normalization status: {}".format(args.normalize_reward))
        try:
            print("mean: {}, std {}".format(self.reward_model.mean, self.reward_model.std))
        except:
            print("reward model has no mean")


        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.output_key = args.output_key

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        '''
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        '''
        
        #logger.info(f"queries[0]: {queries[0]}")
        #print(f"queries[0]: {queries[0]}")
        #print(self.tokenizer.encode(queries[0]))

        token_pattern = [128009, 128006, 78191, 128007, 271]

        # 128009, 128006, 78191, 128007, 1432

        


        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                
                for qtoks in inputs["input_ids"]:
                    qtoks = qtoks.tolist()
                    token_pattern_len = len(token_pattern)
                    search_end = len(qtoks)
                    found = False
                    for j in range(search_end - token_pattern_len, -1, -1):
                        if qtoks[j:j + token_pattern_len] == token_pattern:
                            found = True 
                            break
                    if not found:
                        print("token pattern not found, skip example", token_pattern)
                        print(self.tokenizer.decode(qtoks))
                        print(qtoks)
                        #exit()
                


                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                #print(r)
                if self.output_key:
                    r = getattr(r, self.output_key)
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--output_key", type=str, default=None)
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if "ArmoRM" in args.reward_pretrain:
        cls_class = AutoModelForSequenceClassification
    else:
        cls_class = None

    reward_model = RewardModelProxy(args, cls_class)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")