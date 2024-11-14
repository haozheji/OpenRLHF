import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import math 

from typing import Optional, Tuple

from .utils import masked_mean



class KL_Variance(nn.Module):
    def __init__(self,
        alpha: float = 0.9,
        clip_eps: float = 0.2,
        eps=1e-8):

        super().__init__()
        self.alpha = alpha 
        self.clip_eps = clip_eps
        self.eps = eps 
    
        self.step = 1 

        self.register_buffer("p", torch.zeros(1))

    def update_probs(self, log_probs):
        device = log_probs.device 
        ret_dtype = log_probs.dtype

        self.p = self.alpha * self.p.to(device=device, dtype=ret_dtype) + (1 - self.alpha) * log_probs.exp()
        self.step += 1
    
    def reset_probs(self):
        self.step = 1
        self.p = torch.zeros(1)


    def forward(self, log_probs, old_log_probs, action_mask):
        ratio = (log_probs - old_log_probs).exp()

        device = ratio.device 
        ret_dtype = ratio.dtype

        probs_average = self.p / (1 - torch.tensor(self.alpha, device=device, dtype=ret_dtype).pow(self.step))
        kl_variance = masked_mean(ratio * (log_probs - probs_average.clamp(min=self.eps).log()), action_mask, dim=-1).detach().mean()

        return kl_variance


        
    



class SNR(nn.Module):

    def __init__(
        self, 
        alpha_s: float = 0.9,
        alpha_n: float = 0.9,
        alpha_l: float = 0.9,
        clip_eps: float = 0.2,
        eps=1e-8
    ):
        super().__init__()
        self.alpha_s = alpha_s 
        self.alpha_n = alpha_n
        self.alpha_l = alpha_l
        self.clip_eps = clip_eps
        self.eps = eps

        self.global_step = 1
        self.local_step = 1

        self.register_buffer("s", torch.zeros(1))
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("l", torch.zeros(1))
    
    def update_signal(self, log_probs, old_log_probs, action_mask):
        ratio = (log_probs - old_log_probs).exp()
        entropy = masked_mean(- ratio * log_probs, action_mask, dim=-1).detach().mean()

        #dtype = torch.float32
        device = entropy.device
        ret_dtype = entropy.dtype
        

        self.s = self.alpha_s * self.s.to(device=device, dtype=ret_dtype) + (1 - self.alpha_s) * entropy
        return self.s / (1.0 - torch.tensor(self.alpha_s, device=device, dtype=ret_dtype).pow(self.global_step))
    
    def update_noise(self, log_probs, old_log_probs, action_mask):
        ratio = (log_probs - old_log_probs).exp()

        device = ratio.device 
        ret_dtype = ratio.dtype

        logprobs_average = self.l / (1 - torch.tensor(self.alpha_l, device=device, dtype=ret_dtype).pow(self.local_step))
        entropy_average = masked_mean(-  ratio * logprobs_average, action_mask, dim=-1).detach().mean()

        self.n = self.alpha_n * self.n.to(device=device, dtype=ret_dtype) + (1 - self.alpha_n) * entropy_average
        return self.n / (1.0 - torch.tensor(self.alpha_n, device=device, dtype=ret_dtype).pow(self.global_step))

    def update_logprobs(self, log_probs):
        device = log_probs.device 
        ret_dtype = log_probs.dtype

        self.l = self.alpha_l * self.l.to(device=device, dtype=ret_dtype) + (1 - self.alpha_l) * log_probs

    
    def update_global_step(self):
        self.global_step += 1
    
    def update_local_step(self):
        self.local_step += 1
    
    def reset_logprobs(self):
        self.local_step = 1
        self.l = torch.zeros(1)


    def forward(
        self,
        update_type: float,
        log_probs: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        if update_type == "s":
            return self.update_signal(log_probs, old_log_probs, action_mask)
            
        if update_type == "n":
            return self.update_noise(log_probs, old_log_probs, action_mask)
        
        if update_type == "l":
            self.update_logprobs(log_probs)


    '''
        ratio = (log_probs - old_log_probs).exp()
        kl = (ratio - 1).clamp(- self.clip_eps,  self.clip_eps).pow(2)
        #kl = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * (ratio - 1).clamp(- self.clip_eps, self.clip_eps)
        kl = masked_mean(kl, action_mask, dim=-1).detach() # first average over length, then average over batch
        
        dt = kl.mean() * constant
        
        #print("mt device: ", self.mt.device)
        #print("kl device: ", kl.device)
        #print("step device: ", self.step.device)
        dtype = torch.float32
        ret_dtype = dt.dtype

        mt_weight = 1.0 / (1.0 - torch.tensor(self.alpha1, device=dt.device, dtype=dtype).pow(self.step.to(device=dt.device, dtype=dtype)))
        vt_weight = 1.0 / (1.0 - torch.tensor(self.alpha2, device=dt.device, dtype=dtype).pow(self.step.to(device=dt.device, dtype=dtype)))
        self.mt = self.alpha1 * self.mt.to(device=dt.device, dtype=dtype) + (1 - self.alpha1) * dt.to(dtype)
        self.vt = self.alpha2 * self.vt.to(device=dt.device, dtype=dtype) + (1 - self.alpha2) * dt.to(dtype).pow(2)

        mt_corrected = self.mt.to(ret_dtype) * mt_weight
        vt_corrected = self.vt.to(ret_dtype) * vt_weight
        return mt_corrected, vt_corrected, ((mt_corrected + self.eps) / (vt_corrected.pow(0.5) + self.eps)).to(ret_dtype), kl
    
    def update(self):
        self.step += 1
    '''



