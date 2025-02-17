import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from helper.util import extract
from helper.beta_generator import BetaGenerator
from abc import ABC, abstractmethod

class BaseSampler(nn.Module, ABC):
    def __init__(self, config_path : str):
        super().__init__()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)['sampler']
        self.T = self.config['T']
        self.sampling_T = self.config['sampling_T']
        beta_generator = BetaGenerator(T=self.T)
        
        step = self.T // self.sampling_T
        self.register_buffer('timesteps', 
                             torch.arange(0, self.T, step, dtype=torch.int))
        self.register_buffer('beta', getattr(beta_generator,
                                              f"{self.config['beta']}_beta_schedule",
                                              beta_generator.linear_beta_schedule)())

        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_sqrt', self.alpha.sqrt())
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim = 0))
        self.alpha_bar = self.alpha_bar[self.timesteps]
        self.register_buffer('sqrt_one_minus_alpha_bar', (1. - self.alpha_bar).sqrt())
        self.register_buffer('alpha_bar_prev',
                             torch.cat([self.alpha_bar[0:1], self.alpha_bar[self.timesteps[:-1]]]))
        self.register_buffer('sigma' , None) # should be implemented in the derived class

    @abstractmethod
    @torch.no_grad()
    def get_x_prev(self, x, t, idx, eps_hat):
        pass
        
    def set_network(self, network : nn.Module):
        self.network = network
        
    def q_sample(self, x0, t, eps = None):
        alpha_t_bar = extract(self.alpha_bars, t, x0.shape)
        if eps is None:
            eps = torch.randn_like(x0)
        q_xt_x0 = alpha_t_bar.sqrt() * x0 + (1 - alpha_t_bar).sqrt() * eps
        return q_xt_x0

    @torch.no_grad()
    def reverse_process(self, x_T, only_last=False, **kwargs):
        x = x_T
        if only_last:
            for i, t in tqdm(enumerate(reversed(self.timesteps))):
                idx = len(self.timesteps) - i - 1
                x = self.p_sample(x, t, idx, **kwargs)
            return x
        else:
            x_seq = []
            x_seq.append(x)
            for i, t in tqdm(enumerate(reversed(self.timesteps))):
                idx = len(self.timesteps) - i - 1
                x_seq.append(self.p_sample(x_seq[-1], t, idx, **kwargs))
            return x_seq
        
    @torch.no_grad()
    def p_sample(self, x, t, idx, **kwargs):
        eps_hat = self.network(x = x, t = t, **kwargs)
        x = self.get_x_prev(x, idx, eps_hat)
        return x

    @torch.no_grad()
    def forward(self, x_T, **kwargs):
        return self.reverse_process(x_T, **kwargs)