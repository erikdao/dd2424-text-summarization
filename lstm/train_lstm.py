"""
Main entry to train LSTM summary model
"""
import os
import typing

import pytorch_lightning as pl

from .model import LSTMSummary

def main():
    model = LSTMSummary()
    print(model)


if __name__ == '__main__':
    main()
