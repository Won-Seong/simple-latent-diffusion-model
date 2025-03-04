import torch
import torch.nn as nn
import yaml
from helper.cond_encoder import ConditionEncoder

class UnetWrapper(nn.Module):
    def __init__(self, Unet: nn.Module, config_path: str,
                 cond_encoder = None):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)['unet']
        self.add_module('network', Unet(**config))

        # ConditionalEncoder
        self.add_module('cond_encoder', cond_encoder)
        
    def forward(self, x, t, y=None):
        if t.dim() == 0:
            t = x.new_full((x.size(0), ), t, dtype = torch.int, device = x.device)
        if y is not None:
            assert self.cond_encoder is not None, 'You need to set ConditionalEncoder for conditional sampling.'
            if isinstance(y, str):
                y = self.cond_encoder(y)
                if y.size(0) != x.size(0):
                    y = y.repeat(x.size(0), 1)
            elif torch.is_tensor(y) == False:
                y = torch.tensor([y], device=x.device)
            if y.size(0) != x.size(0):
                y = y.repeat(x.size(0), 1)
                y = self.cond_encoder(y).squeeze()
            return self.network(x, t, y)
        else: 
            return self.network(x, t)