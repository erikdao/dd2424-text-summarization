"""
LSTM-based text summarization models with attention
"""
import os
from typing import Any, List, Optional

import torch
import torch.nn as nn

import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self):
        pass
