import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from transformer.positional_encoding import PositionalEncoding
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

import math
from collections import Counter

import io
import time

from utils.glove_embedding import *
import torch.nn.functional as F

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

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
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.embed_src = GloveEmbedding(embeddings, d_model, word2index)
        self.embed_tgt = GloveEmbedding(embeddings, d_model, word2index)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.embed_src.forward((src))
        tgt_emb = self.embed_tgt.forward((tgt))
        src_emb = self.sequence_to_first_dimension(src_emb)
        tgt_emb = self.sequence_to_first_dimension(tgt_emb)
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.pos_enc(tgt_emb)
        
        print("encoding...")
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        print("decoding...")
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        print("generate output...")
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer_encoder(self.positional_encoding(self.embed_src(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer_decoder(self.positional_encoding(self.embed_tgt(tgt)), memory, tgt_mask)

    def sequence_to_first_dimension(self, tensor, batch=None):
        assert batch is not None
        assert tensor.shape[0] == batch.bs
        return tensor.transpose(0, 1).contiguous()

    def bs_to_first_dimension(self, tensor, batch=None):
        assert batch is not None
        assert tensor.shape[1] == batch.bs
        return tensor.transpose(0, 1).contiguous()




# helper function
def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
