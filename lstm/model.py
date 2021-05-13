"""
LSTM model
"""
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl

pl.seed_everything(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSTMGenerator(pl.LightningModule):
    def __init__(self, input_size=256, hidden_size=256):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
    
    def forward(self, input_batch):
        pass
