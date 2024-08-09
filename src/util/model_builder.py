import torch
from torch import nn

from src.models.strats import Strats


class ModelBuilder:
    
    
    def __init__(self, config) -> None:
        ...
    
    def set_config(self, config) -> None:
        ...
        # validate config
    
    def build(self) -> nn.Module:
        model: nn.Module
        if self.config.model_type == 'strats':
            model = Strats(self.config)
        else:
            raise NotImplementedError('Model type not implemented')
        
        if self.config.checkpoint_path:
            model.load_state_dict(torch.load(self.config.checkpoint_path))
            
        if self.config.device:
            model.to(self.config.device)
            
        if self.config.compile:
            # Doesn't this require optimizer?
            model = torch.compile(model)
        
        return model
