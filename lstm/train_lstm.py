"""
Main entry to train LSTM summary model
"""
import os
import typing

import pytorch_lightning as pl

from logger import logger
from model_v2 import LSTMSummary
from dataloader import create_dataloader
from config import config
from glove import glove


def main():
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'amazon-product-reviews')
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    logger.info("Loading train data...")
    train_loader = create_dataloader(csv_file=train_csv, shuffle=True)

    logger.info("Loading test data...")
    test_loader = create_dataloader(csv_file=test_csv, shuffle=False)

    vocab_size = len(glove.stoi.keys())
    model = LSTMSummary(input_size=vocab_size, hidden_size=config.HIDDEN_SIZE, output_size=vocab_size)
    print(model)

    for idx, data in enumerate(train_loader):
        model.forward(data)
        if idx > 0:
            break


if __name__ == '__main__':
    main()
