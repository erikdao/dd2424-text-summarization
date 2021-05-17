"""
A custom Pytorch Dataloader for Reviews Dataset that has 
already been tokenized to match glove embeddings
"""
import os
import typing
import pandas as pd

import torch
import torch.nn as nn
from torch.utils import data as data_utils

from preprocess.glove import *
from utils.pickle import *

MAX_SENTENCE_LEN = 128

class ReviewDatasetGlove(data_utils.Dataset):
    """
    Class to present the Review Dataset
    using Glove Embeddings
    """
    def __init__(self, mappings, w2i, embed_file, transform=None) :
        self.w2i = w2i
        self.mappings = mappings
        self.embed_dim, self.embeddings = load_glove_embeddings(embed_file)
        inputs = mappings['inputs']
        labels = mappings['labels']
        inputs_idxs = [[toktup[1] for toktup in in_sentence] for in_sentence in inputs]
        inputs_idxs = torch.IntTensor(inputs_idxs) 
        label_idxs = [[toktup[1] for toktup in in_sentence] for in_sentence in labels]    
        label_idxs = torch.IntTensor(label_idxs) 
        self.n_points, self.max_len = label_idxs.shape
        self.data = {"inputs": inputs_idxs, "labels": label_idxs}
        self.transform = transform

    def __len__(self):
        return self.n_points
    
    def __getitem__(self, idx):
        """
        Get inputs and labels (tokens up to self.max_len) for choosen indexes
        To embed the data, load glove into nn.Embedding instead
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.data['inputs'][idx,:]
        labels = self.data['labels'][idx,:]
        sample = {'input': inputs, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloader_glove(mappings, word2index, embed_file, batch_size = 100, 
        shuffle = True):
    """
    Create data loader that returns data as torch IntTensor with glvoe indexes
    To embed the data, load glove into nn.Embedding instead
    """
    
    dataset = ReviewDatasetGlove(
        mappings = mappings,
        w2i = word2index,
        embed_file = embed_file,
        transform=None
    )
    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return loader
    
