"""
Main entry to train LSTM summary model
"""
import os
import typing

import pytorch_lightning as pl

from lstm.logger import logger
# from lstm.model import Decoder, LSTMSummary, Encoder, Decoder
from lstm.model_v2 import LSTMSummary

from dataload.dataloader import create_dataloader_glove
from utils.pickle import pickle_load
from utils.glove_embedding import load_glove_embeddings

EMBEDDING_FILE = '../preprocess/glove.6B.50d.txt'


def main():
    input_dir_tok = "./tokenized-padded"
    input_dir_w2i = "./tokenized"
    logger.info("Loading mappings...")
    mappings_pickle = input_dir_tok + "/tokenized-padded"
    mappings = pickle_load(mappings_pickle)
    sequence_length = len(mappings['inputs'][0])
    logger.debug(f"seq length {sequence_length}")

    logger.info("Loading word2index...")
    word2index_pickle = input_dir_w2i + "/word2index"
    word2index = pickle_load(word2index_pickle)

    logger.info("Loading train loader...")
    train_loader = create_dataloader_glove(
        mappings = mappings,
        word2index = word2index,
        embed_file = EMBEDDING_FILE,
        batch_size = 100,
        shuffle=True
    )

    logger.info("Load embeddings...")
    # word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)

    input_size = len(word2index.keys())
    hidden_size = 512
    output_size = len(word2index.keys())

    model = LSTMSummary(input_size, hidden_size, output_size)
    print(model)

    for idx, data in enumerate(train_loader):
        input, label = data['input'], data['label']

        outputs = model.forward(input)
        import pdb; pdb.set_trace();
        if idx > 1:
            break


if __name__ == '__main__':
    main()
