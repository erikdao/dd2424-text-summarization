from preprocess.glove import *
import torch
import torch.nn as nn
import math

class GloveEmbedding():
    def __init__(self, embeddings, w2i):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2i = w2i
        self.vocabulary_size = len(self.w2i.keys())
        self.word_emb_size = 50
        
        self.trained_emb = nn.Embedding(self.vocabulary_size, self.word_emb_size).to(device)
        self.trained_emb.weight.data.uniform_(-0.1, 0.1)

        self.word_emb = nn.Embedding(self.vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings).to(device), requires_grad=False)

    def forward(self, idxs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idxs = idxs.to(device) 
        return (torch.cat((self.trained_emb(idxs), self.word_emb(idxs)), -1) * math.sqrt(2*self.word_emb_size)).to(device)
