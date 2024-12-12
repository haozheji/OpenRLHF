import argparse
import os
from datetime import timedelta

import jsonlines
import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from openrlhf.datasets import PromptDataset, SFTDataset, RewardDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import blending_datasets, get_processor, get_strategy, get_tokenizer

from datasets import load_dataset

from utils.data_utils import save_parquet, save_json, gather_data_distributed

def batch_generate_vllm(args):
    from vllm import LLM, SamplingParams

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        use_beam_search=False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )

    '''
    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    '''
    prompts_data = load_dataset(args.dataset, split=args.dataset_split)

    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, dummy_strategy, input_template=args.input_template, max_length=args.prompt_max_len)
    prompts = list(prompts_dataset)

    # Conditional SFT inference
    if args.enable_csft:
        for i in range(len(prompts)):
            prompts[i] += args.csft_prompt.strip() + " "

    # best of n
    N = args.best_of_n
    output_dataset = []

    outputs = llm.generate(prompts * N, sampling_params)
    for output in outputs:
        prompt = output.prompt
        output = output.outputs[0].text
        output_dataset.append({"input": prompt, "output": output})

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(output_dataset)




def batch_generate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=720))

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}
    '''
    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    '''
    prompts_data = load_dataset(args.dataset, split=args.dataset_split)
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.micro_batch_size, True, False, drop_last=False
    )
    pbar = tqdm(
        prompts_dataloader,
        desc="Generating",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()
    N = args.best_of_n
    output_dataset = []

    for prompts in pbar:
        # Conditional SFT inference
        if args.enable_csft:
            for i in range(len(prompts)):
                prompts[i] += args.csft_prompt.strip() + " "

        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy_sampling,
                top_p=args.top_p,
                early_stopping=True,
                num_beams=1,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt) :]
                output_dataset.append({"input": prompt, "output": output})

        dist.barrier()

    with jsonlines.open(args.output_path + str(strategy.get_rank()), mode="w") as writer:
        writer.write_all(output_dataset)

    # wait unitl all processes generate done
    dist.barrier()

    # concate multiple output files in rank 0
    if strategy.is_rank_0():
        output_dataset = []
        world_size = dist.get_world_size()
        files = [args.output_path + str(rank) for rank in range(world_size)]
        for file in files:
            with jsonlines.open(file, mode="r") as reader:
                for obj in reader:
                    output_dataset.append(obj)
            os.remove(file)

        with jsonlines.open(args.output_path, mode="w") as writer:
            writer.write_all(output_dataset)


def concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """

    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            # left pad
            return torch.cat(
                [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
            )

    max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
    inputs_ids = torch.cat(
        (
            pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
        ),
        dim=0,
    )
    max_length = max(c_mask.shape[1], r_mask.shape[1])
    att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
    return inputs_ids, att_masks



def batch_rm_pair_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    cls_class = None
    if args.model_type == "ArmoRM":
        cls_class = AutoModelForSequenceClassification

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        cls_class=cls_class,
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # prepare models
    model = strategy.prepare(model)
    model.eval()

    raw_dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = RewardDataset(raw_dataset, tokenizer, args.max_len, strategy, input_template=args.input_template, output_template=args.output_template)

    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        desc="Rewarding",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    output_dataset = []
    ids_count = 0 
    win_count = 0 

    with torch.no_grad():
        for chosen_ids, chosen_masks, reject_ids, rejects_masks, extras, ids in pbar:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            chosen_masks = chosen_masks.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            rejects_masks = rejects_masks.squeeze(1).to(torch.cuda.current_device())

            
            
            input_ids, attention_masks = concatenated_inputs(tokenizer, chosen_ids, chosen_masks, reject_ids, rejects_masks)

            
            if args.model_type == "ArmoRM":
                rewards = model(input_ids, attention_masks).score
            else:
                rewards = model(input_ids, attention_masks)

            #print(rewards.size(), ids)

            
            chosen_rewards = rewards[: chosen_ids.shape[0]]
            rejected_rewards = rewards[chosen_ids.shape[0] :]

            for i, chosen_win in enumerate(chosen_rewards > rejected_rewards):
                if chosen_win.item():
                    chosen = raw_dataset[ids[i]][args.chosen_key]
                    reject = raw_dataset[ids[i]][args.rejected_key]
                    chosen_reward = chosen_rewards[i].item()
                    rejected_reward = rejected_rewards[i].item()
                else:
                    chosen = raw_dataset[ids[i]][args.rejected_key]
                    reject = raw_dataset[ids[i]][args.chosen_key]
                    chosen_reward = rejected_rewards[i].item()
                    rejected_reward = chosen_rewards[i].item()
                
                output_dataset.append({args.prompt_key: raw_dataset[ids[i]][args.prompt_key], 
                                       args.chosen_key: chosen,
                                       args.rejected_key: reject,
                                       "chosen_reward": chosen_reward,
                                       "rejected_reward": rejected_reward})
                
                #accs.append(float(chosen_win.item()))
                win_count += chosen_win.long()

            dist.barrier()

            ids_count += chosen_ids.shape[0]
            #print(chosen_ids.shape[0], ids_count)
            
    
    # reduce to rank 0
    ids_count = torch.tensor([ids_count], dtype=win_count.dtype).to(win_count.device)
    dist.all_reduce(ids_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(win_count, op=dist.ReduceOp.SUM)


    output_dataset = gather_data_distributed(output_dataset, args.output_path, strategy.get_rank(), total_num=len(raw_dataset), del_cache=False)


    if strategy.is_rank_0():
        print(f"Total number of samples: {len(output_dataset)}")
        #print(win_count)
        #print(ids_count)
        print(f"Accuracy: {win_count.item() / ids_count.item()} ({win_count.item()} / {ids_count.item()})")
        #rewards = torch.tensor([obj["reward"] for obj in output_dataset])
        #print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        print(output_dataset[0]["instruction"])
        print(output_dataset[1]["instruction"])
        save_split = args.dataset_split.split("[")[0]
        save_json(output_dataset, args.output_path, save_split)
        #with jsonlines.open(args.output_path, mode="w") as writer:
        #    writer.write_all(output_dataset)






def batch_rm_inference(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(minutes=180))

    cls_class = None
    if args.model_type == "ArmoRM":
        cls_class = AutoModelForSequenceClassification
    
    

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        cls_class=cls_class,
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        value_head_prefix=args.value_head_prefix,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)


    
    # prepare models
    model = strategy.prepare(model)
    model.eval()
    '''
    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        return_eval=False,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )
    '''
    #dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = SFTDataset(
        dataset, tokenizer, args.max_len, strategy, pretrain_mode=False, input_template=args.input_template, prompt_max_length=args.prompt_max_len
    )
    #print(dataset[0])
    #print(tokenizer.decode(dataset[0][1][0]))
    #print(dataset.raw_prompts[0])
    #exit()
    dataloader = strategy.setup_dataloader(
        dataset, args.micro_batch_size, True, False, dataset.collate_fn, drop_last=False
    )
    pbar = tqdm(
        dataloader,
        desc="Rewarding",
        disable=not strategy.is_rank_0(),
    )

    dist.barrier()

    output_dataset = []
    with torch.no_grad():
        for _, input_ids, attention_masks, info in pbar:
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            attention_masks = attention_masks.squeeze(1).to(torch.cuda.current_device())

            
            if args.model_type == "ArmoRM":
                rewards = model(input_ids, attention_masks).score
            else:
                rewards = model(input_ids, attention_masks)

            for prompt, output, reward in zip(info["raw_input"], info["output"], rewards):
                output_dataset.append({args.input_key: prompt, args.output_key: output, "reward": reward.item()})

            dist.barrier()
    
    output_dataset = gather_data_distributed(output_dataset, args.output_path, strategy.get_rank(), total_num=len(dataset), del_cache=False)


    # concate multiple output files in rank 0
    if strategy.is_rank_0():

        rewards = torch.tensor([obj["reward"] for obj in output_dataset])
        print(f"Reward mean: {rewards.mean().item()}, std: {rewards.std().item()}")

        if args.post_processor and args.post_processor != "null":
            strategy.print(f"Use Processor {args.post_processor}, Reward Norm {args.normalize_reward}")
            processor = get_processor(args.post_processor)
            output_dataset = processor(args, output_dataset, input_key=args.input_key, output_key=args.output_key)

        save_split = args.dataset_split.split("[")[0]
        save_json(output_dataset, args.output_path, save_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=128, help="Prerequisite for initializing deepspeed. Should match world size")
    parser.add_argument("--seed", type=int, default=1234)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, help="Handle special model type"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=argparse.FileType('r'), default=None)
    parser.add_argument("--output_template", type=argparse.FileType('r'), default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )
    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    args = parser.parse_args()
    if args.input_template is not None:
        args.input_template = args.input_template.read()
    if args.output_template is not None:
        args.output_template = args.output_template.read()

    
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    if args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    elif args.eval_task and args.eval_task == "rm_pair":
        batch_rm_pair_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
