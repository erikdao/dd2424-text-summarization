import os
import pandas as pd
import torch
import spacy
import numpy as np

def load_glove_embeddings(embedding_file):
    """
    Load GloVe word embeddings
    embedding_file: name of txt file containing GloVe word embeddings
    return:
    (vocab size, vector dimension, embedding matrix, word to index mapping)
    """
    padding_idx = 0
    padding_word = '<PAD>'
    unknown_word = '<UNK>'

    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for _,line in enumerate(f):
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])

    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 1
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1

    return N, D, np.array(embeddings, dtype=np.float32), word2index

