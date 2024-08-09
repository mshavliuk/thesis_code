from dataclasses import dataclass
from itertools import (
    chain,
    repeat,
)

import torch
from torch import nn
from tqdm import tqdm

from src.util.data_loader import DataLoader

@dataclass(frozen=True)
class TrainerConfig:
    train_batch_size: int
    eval_batch_size: int
    lr: float
    max_epochs: int
    

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        loss = (pred - target)[mask].pow(2).mean()
        return loss

class Trainer:
    def __init__(self, logger):
        self.logger = logger
        
        
    def train(self, config:TrainerConfig, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader=None):
        # model: nn.Module = torch.compile(model)
        # assert model.device == train_loader.device, "Model and DataLoader should be on the same device"
        
        criterion = MaskedMSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        # todo: try bf16
        torch.set_float32_matmul_precision('high')
        
        evaluator = Evaluator(self.logger, val_loader, criterion)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.max_epochs):
            train_loss = 0.0
            model.train()
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                optimizer.zero_grad()
                output = model(**batch.model_inputs)
                
                loss = criterion(output, batch['forecast_values'], batch['forecast_mask'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.logger.info(f'Epoch {epoch + 1}, Loss: {train_loss}')
            # Validation step (optional)
            # if epoch % validation_interval == 0:
            val_loss = evaluator.evaluate(model)
            
            
            
            # # TODO: early stopping
            # model.evaluate(train_loader)  # TODO: use separate evaluator class that would report to wandb
            # model.checkpoint()
        model.save()
    

class Evaluator:
    def __init__(self, logger, val_loader: DataLoader, criterion: nn.Module):
        self.logger = logger
        self.val_loader = val_loader
        self.criterion = criterion
        self.evaluation_batches = []
        
        
    def evaluate(self, model: nn.Module):
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        
        if len(self.evaluation_batches) == 0:
            for batch in tqdm(chain(*[self.val_loader] * 3), desc='Generating validation data'):
                self.evaluation_batches.append(batch)

        with torch.no_grad():
            for batch in tqdm(self.evaluation_batches, desc='Validation'):
                outputs = model(**batch.model_inputs)
                loss = self.criterion(outputs, batch['forecast_values'], batch['forecast_mask'])
                val_loss += loss.item()
        val_loss /= len(self.evaluation_batches)
        self.logger.info(f'Validation Loss: {val_loss}')
        return val_loss
