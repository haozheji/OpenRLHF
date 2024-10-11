import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models import GPTLMLoss
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean, compute_approx_kl
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from openrlhf.datasets.utils import alter_template_texts

logger = init_logger(__name__)


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, 
                oracle_reward_model: nn.Module = None, 
                remote_oracle_url: str = None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.oracle_reward_model = oracle_reward_model
        self.remote_oracle_url = remote_oracle_url
    
    @torch.no_grad()
    def eval_constraint(self, dataloader, global_step, **generate_kwargs):
        self.actor.eval()
        device = torch.cuda.current_device()


        pbar = tqdm(dataloader,
                    desc=f"Evaluating constraint @{global_step} step",
                    disable=not self.strategy.is_rank_0(),)
        
        
        mle_fn = GPTLMLoss()

        seq_nlls = []
        seq_kls = []
        for (prompt_lens, input_ids, _, info) in pbar:

            # generate sequence
            prompts = info["input"]

            sequences, attention_mask, action_mask, output_texts = (
                self._generate_local(prompts, **generate_kwargs)
                if self.vllm_engines is None
                else self._generate_vllm(prompts, **generate_kwargs)
            )

            num_actions = action_mask.size(1)
            sequences_cpu, attention_mask_cpu, action_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
                action_mask.to("cpu"),
            )

            # init action log probs
            if self.strategy.args.colocate_actor_ref:
                torch.cuda.empty_cache()
                ray.get([self.initial_model.empty_cache.remote()])

            base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

            base_action_log_probs = ray.get(base_action_log_probs_ref).to(device)
            if self.strategy.args.colocate_actor_ref:
                ray.get([self.initial_model.empty_cache.remote()])

            # actor action log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask)
            
            seq_kl = compute_approx_kl(action_log_probs, base_action_log_probs, action_mask).sum(-1)



            # calculate action mle loss
            input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
            
            input_ids, attention_mask, action_mask = self.actor.process_sequences(
                input_ids, max(prompt_lens), self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
            )

            # prepare action mask
            num_actions = action_mask.size(1)
            action_log_probs = self.actor(input_ids, num_actions, attention_mask)

            seq_nll = - (action_log_probs * action_mask).sum(-1)

            seq_nlls.append(seq_nll)
            seq_kls.append(seq_kl)
        
        self.actor.train()  # reset model state

        seq_nlls = torch.cat(seq_nlls, dim=0)
        seq_kls = torch.cat(seq_kls, dim=0)

        return seq_kls.mean(), seq_nlls.mean()








            


    
    @torch.no_grad()
    def eval_oracle_reward(self, dataloader, global_step, **generate_kwargs):
        pbar = tqdm(dataloader,
                    desc=f"Evaluating oracle reward @{global_step} step",
                    disable=not self.strategy.is_rank_0(),)

        if "ArmoRM" in self.strategy.args.oracle_pretrain:
            output_key = "score"
        else:
            output_key = None

        refs = []
        for (eval_prompts, raw_prompts) in pbar:
            self.strategy.print(eval_prompts[0], len(eval_prompts))
            self.strategy.print(raw_prompts[0], len(raw_prompts))
            oracle_reward = self.eval_oracle_reward_batch(eval_prompts, raw_prompts, output_key=output_key, **generate_kwargs)
            refs.append(oracle_reward)
        return refs 

    def gather_oracle_reward(self, refs):
        refs = ray.get(refs)
        oracle_reward_mean = 0
        count = 0 
        for oracle_reward in refs:
            oracle_reward_mean += oracle_reward.sum().item()
            count += oracle_reward.numel()
        oracle_reward_mean /= count
        return oracle_reward_mean
    
    
    @torch.no_grad()
    def eval_oracle_reward_batch(self, prompts: Union[str, List[str]], raw_prompts, output_key: str = None, **generate_kwargs):
        #self.actor.eval()
        device = torch.cuda.current_device()

        if self.oracle_reward_model is None and not self.remote_oracle_url:
            raise RuntimeError("Try to evaluate oracle reward, but oracle reward model is None!")

        # generate sequence
        start = time.time()
        sequences, attention_mask, action_mask, output_texts = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        # pre-truncate input and output 
        texts = []
        for raw_prompt, output_text in zip(raw_prompts, output_texts):
            trunc_prompt = self.tokenizer.decode(self.tokenizer.encode(raw_prompt)[:self.strategy.args.prompt_max_len], skip_special_tokens=True).strip()
            trunc_output = self.tokenizer.decode(self.tokenizer.encode(output_text)[:self.strategy.args.generate_max_len], skip_special_tokens=True).strip()
            texts.append(self.strategy.args.oracle_template.format(trunc_prompt, trunc_output))

        #texts = [self.strategy.args.oracle_template.format(raw_prompt, output_text) for raw_prompt, output_text in zip(raw_prompts, output_texts)]

        # evaluate oracle reward
        if not self.remote_oracle_url:
            tokenizer = ray.get(self.oracle_reward_model.get_tokenizer.remote()),
            inputs = tokenizer(texts, 
                                max_length=self.strategy.args.prompt_max_len + self.strategy.args.generate_max_len,
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt",
                                add_special_tokens=True)
            
            oracle_reward = self.oracle_reward_model.forward.remote(inputs["input_ids"], inputs["attention_mask"], output_key)

        else:
            oracle_reward = remote_rm_fn_ray.remote(self.remote_oracle_url, queries=texts)
        
        return oracle_reward


    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        sequences, attention_mask, action_mask, texts = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        # init log probs
        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()
            ray.get([self.initial_model.empty_cache.remote()])
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # values
        value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)


        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        output_texts = []
        for prompt, output in zip(prompts, outputs):
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            if output_ids[output_len - 1] != eos_token_id:
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)
            output_texts.append(self.tokenizer.decode(output_ids, skip_special_tokens=True))

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda"), output_texts

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
