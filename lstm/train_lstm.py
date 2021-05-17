"""
Main entry to train LSTM summary model
"""
import os
import typing

import pytorch_lightning as pl

from lstm.logger import logger
from lstm.model import Decoder, LSTMSummary, Encoder, Decoder

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
    word_emb_size, embeddings = load_glove_embeddings(EMBEDDING_FILE)
    
    encoder = Encoder(embeddings=embeddings, embedding_dim=word_emb_size,
                      word2index=word2index, encoder_hidden_dim=256, decoder_hidden_dim=256,
                      dropout=0.5)
    print(encoder)

    decoder = Decoder(embeddings=embeddings, word2index=word2index, embedding_dim=word_emb_size,
                      output_dim=1024, encoder_hidden_dim=256, decocder_hidden_dim=256, dropout=0.5)
    print(decoder)

    for idx, data in enumerate(train_loader):
        input_vec = data['input']
        label_vec = data['label']
        print('label_vec', label_vec.shape)
        encoder_outputs, hidden = encoder.forward(input_vec)
        print('hidden', hidden.shape)


        input_ = label_vec[:, 0]  # First word of the label
        outputs = []
        for i in range(1, label_vec.shape[1]):
            output, hidden = decoder.forward(input_, hidden)
            outputs.append(output)
        
        if idx > 0:
            break

if __name__ == '__main__':
    main()
