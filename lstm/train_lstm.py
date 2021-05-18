"""
Main entry to train LSTM summary model
"""
import os
import typing

import torchtext.vocab as vocab
import pytorch_lightning as pl

pl.seed_everything(42, workers=True)

from logger import logger
from model_v2 import LSTMSummary
from dataloader import create_dataloader
from config import config

from glove import extend_glove

glove = extend_glove(vocab.GloVe(name='6B', dim=50))


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
    
    trainer = pl.Trainer(fast_dev_run=False, max_epochs=config.EPOCHES, gpus=1)
    trainer.fit(model, train_loader, test_loader)
    # for idx, data in enumerate(train_loader):
    #     model.training_step(data)
    #     if idx > 0:
    #         break


if __name__ == '__main__':
    main()
