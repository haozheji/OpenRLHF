import os 
from vllm import SamplingParams, LLM
import multiprocessing
from tqdm import tqdm

from datasets import load_dataset

import argparse

from openrlhf.datasets import PromptDataset

from transformers import AutoTokenizer

from utils.data_utils import save_json


def worker_mp(args):
    return worker(*args)

def worker(worker_idx, args, prompts, raw_prompts):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_idx)
    
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.generate_max_len)

    llm = LLM(model=args.model_dir)

    output_texts = []
    print(f'Process: {os.getpid()} Worker: {worker_idx} Total samples: {len(prompts)} Batch size: {args.micro_batch_size}', flush=True)
    for bidx in tqdm(range(0, len(prompts), args.micro_batch_size), desc=f"Worker {worker_idx}"):
        batch = prompts[bidx:bidx+args.micro_batch_size]
        outputs = llm.generate(batch, sampling_params)
        raw_prompts_batch = raw_prompts[bidx:bidx+args.micro_batch_size]
        for raw_prompt, output in zip(raw_prompts_batch, outputs):
            output_texts.append({"instruction": raw_prompt, "response": output.outputs[0].text})
    
    return output_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default="/workspace/jihaozhe/models/zephyr-7b-sft-full")
    
    # data args
    parser.add_argument("--dataset", type=str, default="/workspace/jihaozhe/data/ultrafeedback-binarized-preferences")
    parser.add_argument("--split", type=str, default="train[-4:]")
    parser.add_argument("--input_key", type=str, default="instruction")
    parser.add_argument("--input_template", type=argparse.FileType('r'), default="src/template/zephyr.in")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    
    # inference args
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--N", type=int, default=2)

    # file args
    parser.add_argument("--save_dir", type=str, default="/workspace/jihaozhe/data/")
    parser.add_argument("--save_split", type=str, default="train")

    args = parser.parse_args()

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    devices = [int(x) for x in args.devices.split(",")]
    if args.input_template is not None:
        args.input_template = args.input_template.read()
    else:
        args.input_template = "{}"
    

    # data preprocessing
    print("Loading dataset")
    prompts_data = load_dataset(args.dataset, split=args.split)
    prompts_dataset = PromptDataset(prompts_data, 
                                    tokenizer, 
                                    dummy_strategy, 
                                    input_template=args.input_template, 
                                    max_length=args.prompt_max_len,
                                    return_raw=True)
    
    # prepare for bon
    repeated_prompts_dataset = []
    raw_prompts_dataset = []
    for p, r in prompts_dataset:
        for i in range(args.N):
            repeated_prompts_dataset.append(p)
            raw_prompts_dataset.append(r)
    
    print("Number of prompts: ", len(repeated_prompts_dataset))
    print("Example prompt: ", repeated_prompts_dataset[0])
    print("="*40)
    num_processes = len(devices)
    samples_per_worker = len(repeated_prompts_dataset) // (num_processes - 1)
    print("Number of samples per worker: ", samples_per_worker)

    jobs = []
    with multiprocessing.Pool(num_processes) as pool:
        mp_args = []
        
        for i, device in enumerate(devices):
            worker_dataset = repeated_prompts_dataset[i*samples_per_worker : (i+1) * samples_per_worker]
            worker_raw_dataset = raw_prompts_dataset[i*samples_per_worker : (i+1) * samples_per_worker]
            mp_args.append((device, args, worker_dataset, worker_raw_dataset))
            
        jobs = pool.map(worker_mp, mp_args)
    
    #print(len(jobs))
    #print(jobs[0])
    results = []
    print(jobs[0][0])
    for job in jobs:
        results.extend(job)
    
    save_json(results, args.save_dir, args.save_split)
    


    
