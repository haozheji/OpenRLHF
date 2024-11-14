import argparse

import ray
import torch 
from torch import distributed as dist
from vllm import SamplingParams

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.utils import get_strategy, get_tokenizer



def generate_ray_vllm(args):
    # configure strategy
    strategy = get_strategy(args)

    # this line setup deepspeed distributed
    
    strategy.setup_distributed()
    print(dist.get_world_size())

    vllm_engines = create_vllm_engines(
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
                args.pretrain,
                args.seed,
                args.enable_prefix_caching,
                args.prompt_max_len + args.generate_max_len,
    )

    # init process group
    if torch.distributed.get_rank() == 0:
        print("Yeah!")

    # round-robin load balance
    
    llm = vllm_engines[rank % len(vllm_engines)]

    print(vllm)

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")


    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)



    args = parser.parse_args()

    generate_ray_vllm(args)