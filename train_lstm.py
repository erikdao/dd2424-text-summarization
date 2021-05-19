"""
Main entry to train LSTM summary model
"""
import os
import typing
import pickle
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(1, workers=True)

from dataload.dataloader import create_dataloader_glove
from lstm.config import config

from lstm.logger import logger
from lstm.model_v2 import LSTMSummary

from utils.glove_embedding import load_glove_embeddings


def pickle_load(filename: str) -> typing.Any:
    "Load python object from a pickle file"
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def main_v2(args: typing.Any) -> None:
    """New experiments with new padded tokens"""
    # Figure out the path to data
    data_dir = os.path.join(os.getcwd(), "data")
    logger.info("Loading mappings...")
    mappings = pickle_load(os.path.join(data_dir, "tokenized-padded"))

    logger.info("Loading word2index...")
    word2index = pickle_load(os.path.join(data_dir, "word2index"))
    vocab_size = len(word2index.keys())

    logger.info("Loading GloVe.6B.50d embeddings...")
    embedding_dim, embeddings = load_glove_embeddings(
        os.path.join(data_dir, "glove6B", "glove.6B.50d.txt")
    )

    # Create data splits
    inputs, labels = mappings["inputs"], mappings["labels"]
    train_mappings = {"inputs": inputs[:327200], "labels": labels[:327200]}
    val_mappings = {
        "inputs": inputs[327200 : 327200 + 40900],
        "labels": labels[327200 : 327200 + 40900],
    }
    test_mappings = {
        "inputs": inputs[327200 + 40900 : 327200 + 2 * 40900],
        "labels": labels[327200 + 40900 : 327200 + 2 * 40900],
    }

    if args.dev:  # Development mode, use less data
        train_mappings = {"inputs": inputs[:8000], "labels": labels[:8000]}
        val_mappings = {
            "inputs": inputs[8000 : 8000 + 1000],
            "labels": labels[8000 : 8000 + 1000],
        }
        test_mappings = {
            "inputs": inputs[8000 + 1000 : 8000 + 2 * 1000],
            "labels": labels[8000 + 1000 : 8000 + 2 * 1000],
        }

    # Create data loaders
    logger.info("Create training loader...")
    train_loader = create_dataloader_glove(
        mappings=train_mappings,
        word2index=word2index,
        embeddings=embeddings,
        word_emb_size=embedding_dim,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    logger.info("Create val loader...")
    val_loader = create_dataloader_glove(
        mappings=val_mappings,
        word2index=word2index,
        embeddings=embeddings,
        word_emb_size=embedding_dim,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    logger.info("Create test loader...")
    test_loader = create_dataloader_glove(
        mappings=test_mappings,
        word2index=word2index,
        embeddings=embeddings,
        word_emb_size=embedding_dim,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    model = LSTMSummary(
        embedding_size=vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        word2index=word2index,
        embeddings=embeddings,
    )
    
    data = next(iter(train_loader))
    model.forward(data["input"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()
    main_v2(args)
