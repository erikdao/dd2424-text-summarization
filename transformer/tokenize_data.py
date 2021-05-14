"""
Main entrance for training LSTM for text summary generation
- The word embedding used is DistillBert
"""
from dataloader import * 
import os
import pandas as pd
import torch
import spacy

CSV_FILE = '../data/amazon-product-reviews/Reviews.csv'
VOCAB_FILE = ''

def main():
    print("Loading data...")
    data = pd.read_csv(CSV_FILE)
    inputs = data['Text'].values
    labels = data['Summary'].values
    print("Loading spacy model...")
    lang_model = spacy.load('en_core_web_sm')

    punctuation = ['(', ')', ':', '"', ' ']

    n_points = len(inputs)
    processed_inputs = []
    processed_labels = []
    
    # split sentences into (reasonable) words
    print("Processing tokens...")
    for i in range(n_points):
        in_sen = inputs[i].lower()
        in_sen = [tok.text for tok in lang_model.tokenizer(in_sen) \
            if tok.text not in punctuation] 
        processed_inputs.append(in_sen)
        lab_sen = labels[i].lower()
        lab_sen = [tok.text for tok in lang_model.tokenizer(lab_sen) \
            if tok.text not in punctuation] 
        processed_labels.append(lab_sen)

    print(processed_inputs)

if __name__ == '__main__':
    main()
