from preprocess.glove import *
import torch
import torch.nn as nn
import math

class GloveEmbedding():
    def __init__(self, embeddings, word_emb_size, w2i):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2i = w2i
        self.vocabulary_size = len(self.w2i.keys())
        self.word_emb_size = word_emb_size

        self.word_emb = nn.Embedding(self.vocabulary_size, self.word_emb_size)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings).to(device), requires_grad=False)
        # TODO maybe we shouldnt freeze or add a trainable part to it

    def forward(self, idxs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idxs = idxs.to(device) 
        return self.word_emb(idxs) * math.sqrt(self.word_emb_size)
