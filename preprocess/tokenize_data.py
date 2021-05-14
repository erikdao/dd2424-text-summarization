"""
Main entrance for training LSTM for text summary generation
- The word embedding used is DistillBert
"""
import os
import pandas as pd
import torch
import spacy
import numpy as np
import pickle
from preprocess.glove import *
from utils.pickle import *

CSV_FILE = '../data/amazon-product-reviews/Reviews.csv'
EMBEDDING_FILE = './glove.6B.50d.txt'
#WORD2INDEX_FILE = 'w2i.pickle'
#TOKENIZED_DATA_FILE = 'tokenized.pickle'

def tokenize_data(inputs, labels, word2index):
    """
    Splits data sentences into reasonable words with spacy
    then map words -> indexes with mapping for glove embeddings.
    Unknown tokens get the same index as the unknown token <UNK>

    inputs: list with input sentences 
    labels: list with label sentences
    word2index: mapping for vocabulary's words to indexes

    return: 
    mappings: a dictionary containing mappings between data's tokens and indexes ->
        {
            'inputs': [[(token, tokenidx),  ...], ...], 
            'labels': [[(token, tokenidx),  ...], ...]
         }
    n_valid_points: number of valid points after spacy tokenizer
    """

    punctuation = ['(', ')', ':', '"', ' ']

    lang_model = spacy.load('en_core_web_sm')
    n_points = len(inputs)
    processed_inputs = []
    processed_labels = []

    mappings = {'inputs': [], 'labels': []}
    
    # split sentences into (reasonable) words
    n_valid_points = 0
    for i in range(n_points):
        in_sen = inputs[i]
        lab_sen = labels[i]

        if type(in_sen) == str and type(lab_sen) == str:
            in_sen = in_sen.lower()
            in_sen = [tok.text for tok in lang_model.tokenizer(in_sen) if tok.text not in punctuation] 
            in_sen_idxs = [(tok, word2index.get(tok, 1)) for tok in in_sen] 
            mappings['inputs'].append(in_sen_idxs)

            lab_sen = lab_sen.lower()
            lab_sen.lower()
            lab_sen = [tok.text for tok in lang_model.tokenizer(lab_sen) if tok.text not in punctuation] 
            lab_sen_idxs = [(tok, word2index.get(tok, 1)) for tok in lab_sen] 
            mappings['labels'].append(in_sen_idxs)
            n_valid_points += 1

    return mappings, n_valid_points


def main():
    print("Loading data...")
    data = pd.read_csv(CSV_FILE)
    inputs = data['Text'].values
    labels = data['Summary'].values
    n_points = len(inputs)
    
    # build indexing
    print("creating word2index and loading embeds...")
    vocab_size, word_emb_size, embeddings, word2index = \
        load_glove_embeddings(embedding_file = EMBEDDING_FILE)

    # save embeddings and word2index dict
    #print("embds: ",embeddings)
    #print("w2i: ",word2index)

    print("Saving word2index to file...")
    pickle_save("./word2index", word2index)

    print("Process inputs to reasonable tokens...")
    mappings, n_valid_points = tokenize_data(inputs,labels, word2index)

    #print("valid points: ", n_valid_points, " / ", n_points)
    #print("maps: ",mappings)
    
    print("Saving token to index mapping to file...")
    pickle_save("./tokenized", mappings)

if __name__ == '__main__':
    main()
