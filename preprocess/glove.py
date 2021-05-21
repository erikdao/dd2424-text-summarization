import os
import pandas as pd
import torch
import spacy
import numpy as np

# This file contains:
# def load_glove_embeddings_wordindex()
# def load_glove_embeddings()

def load_glove_embeddings_wordindex(embedding_file):
    """
    Load GloVe word embeddings
    embedding_file: name of txt file containing GloVe word embeddings
    return:
    (vocab size, vector dimension, embedding matrix, word to index mapping)
    """
    padding_idx = 0
    padding_word = '<PAD>'
    unknown_word = '<UNK>'
    start_word = '<SOS>'
    end_word = '<EOS>'

    word2index, embeddings, N = {}, [], 0
    mean = 0
    std = 0
    with open(embedding_file, encoding='utf8') as f:
        for _,line in enumerate(f):
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            vec_np = np.asarray(vec)
            mean += vec_np.mean()
            std += vec_np.std()
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])
    vocab_size = len(embeddings)
    mean /= vocab_size
    std /= vocab_size

    if padding_idx is not None and type(padding_idx) is int:
        # pad and unkown toks
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        # sos and eos toks
        embeddings.insert(padding_idx + 2, adjust_mean_std(np.asarray(np.random.rand(D), "float32"), mean, std) )
        embeddings.insert(padding_idx + 3, adjust_mean_std(np.asarray(np.random.rand(D), "float32"), mean, std) )

        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 4
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1
        word2index[start_word] = padding_idx + 2
        word2index[end_word] = padding_idx + 3

    return N, D, np.array(embeddings, dtype=np.float32), word2index

def adjust_mean_std(np_distribution, new_mean, new_var):
    """
        Normalizes np_distribution (distribution of floats) to have mean and
        standard deviation new_mean and new_var
    """
    # scale variance
    constant = new_var / np_distribution.std()
    np_distribution *= constant
    # shift mean
    np_distribution = np_distribution - np_distribution.mean() + new_mean
    return np_distribution

def load_glove_embeddings(embedding_file):
    """
    Load GloVe word embeddings
    embedding_file: name of txt file containing GloVe word embeddings
    return:
    (vector dimension, embedding matrix)
    """
    padding_idx = 0
    padding_word = '<PAD>'
    unknown_word = '<UNK>'
    start_word = '<SOS>'
    end_word = '<EOS>'

    embeddings = []
    mean = 0
    std = 0
    with open(embedding_file, encoding='utf8') as f:
        for _,line in enumerate(f):
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            vec_np = np.asarray(vec)
            mean += vec_np.mean()
            std += vec_np.std()
            embeddings.append(vec)
    D = len(embeddings[0])
    vocab_size = len(embeddings)
    mean /= vocab_size
    std /= vocab_size

    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        embeddings.insert(padding_idx + 2, adjust_mean_std(np.asarray(np.random.rand(D), "float32"), mean, std) )
        embeddings.insert(padding_idx + 3, adjust_mean_std(np.asarray(np.random.rand(D), "float32"), mean, std) )

    return D, np.array(embeddings, dtype=np.float32)
