"""
Main entry to train LSTM summary model
"""
import os
import typing
import pickle
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42, workers=True)

from dataload.dataloader import create_dataloader_glove
from lstm.config import config

from lstm.logger import logger
from lstm.model import LSTMSummary

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

    logger.info("Creating index2word...")
    index2word = {v: k for k, v in word2index.items()}

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
        train_mappings = {"inputs": inputs[:10000], "labels": labels[:10000]}
        logger.debug(f"train_mappings - inputs: {len(train_mappings['inputs'])}; labels: {len(train_mappings['labels'])}")
        val_mappings = {
            "inputs": inputs[10000 : 10000 + 1000],
            "labels": labels[10000 : 10000 + 1000],
        }
        logger.debug(f"val_mappings - inputs: {len(val_mappings['inputs'])}; labels: {len(val_mappings['labels'])}")
        test_mappings = {
            "inputs": inputs[10000 + 1000 : 10000 + 2 * 1000],
            "labels": labels[10000 + 1000 : 10000 + 2 * 1000],
        }
        logger.debug(f"test_mappings - inputs: {len(test_mappings['inputs'])}; labels: {len(test_mappings['labels'])}")
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
        index2word=index2word,
        embeddings=embeddings,
    )
    logger.debug(model)

    # Configure checkpoint
    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min")
    # earlystopping_cb = EarlyStopping(monitor="val_loss", patience=8)
    callbacks = [checkpoint_cb]

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.EPOCHES,
        val_check_interval=config.VAL_CHECK_STEP,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        limit_val_batches=0.33
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")

    args = parser.parse_args()
    main_v2(args)
