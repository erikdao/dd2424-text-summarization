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


def main():
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'amazon-product-reviews')
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    logger.info("Loading train data...")
    train_loader = create_dataloader(csv_file=train_csv, shuffle=True)

    for idx, data in enumerate(train_loader):
        input_tensor = data['input']
        output_tensor = data['label']
        print('input_tensor', input_tensor.shape, input_tensor)
        print('output_tensor', output_tensor.shape)

        if idx > 3:
            break

    logger.info("Loading test data...")
    test_loader = create_dataloader(csv_file=test_csv, shuffle=False)
    for idx, data in enumerate(test_loader):
        input_tensor = data['input']
        output_tensor = data['label']
        print('input_tensor', input_tensor.shape)
        print('output_tensor', output_tensor.shape)

        if idx > 3:
            break


if __name__ == '__main__':
    main()
