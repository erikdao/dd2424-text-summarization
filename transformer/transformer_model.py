import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from transformer.positional_encoding import PositionalEncoding

from utils.glove_embedding import *

class SummaryTransformer(nn.Module):
    def __init__(self, 
                vocab_size, 
                d_model, 
                nhead, 
                num_encoder_layers, 
                num_decoder_layers, 
                dim_feedforward, 
                max_seq_length, 
                pos_dropout,
                trans_dropout,
                word2index,
                embeddings):
        super().__init__()
        self.d_model = d_model
        self.embed_src = GloveEmbedding(embeddings, d_model, word2index)
        self.embed_tgt = GloveEmbedding(embeddings, d_model, word2index)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fully_connected = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_mask=None):
        #src = rearrange(src, 'n s -> s n')
        #tgt = rearrange(tgt, 'n t -> t n')
        src = self.pos_enc(self.embed_src.forward((src)) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt.forward((tgt)) * math.sqrt(self.d_model))

        output = self.transformer(src, tgt)# optional, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
        #output = rearrange(output, 't n e -> n t e')
        return self.fully_connected(output)
