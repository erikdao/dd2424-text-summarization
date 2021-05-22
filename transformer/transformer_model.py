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
torch.use_deterministic_algorithms(False)

class SummaryTransformer(nn.Module):
    def __init__(self, 
                vocab_size, 
                word_emb_size, 
                nhead, 
                num_encoder_layers, 
                num_decoder_layers, 
                dim_feedforward, 
                max_input_seq_length,
                max_label_seq_length, 
                pos_dropout,
                trans_dropout,
                word2index,
                embeddings):
        super().__init__()
        self.model_type = 'Transformer'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary_size = vocab_size
        self.word_emb_size = word_emb_size
        self.d_model = self.word_emb_size * 2
        self.embed_src = GloveEmbedding(embeddings, word2index)
        self.embed_tgt = GloveEmbedding(embeddings, word2index)
        
        # add random learnable part to embedding GLOVE
        self.word_emb_extention = nn.Embedding(self.vocabulary_size, self.word_emb_size)
        self.word_emb_extention.weight = nn.Parameter(torch.from_numpy(embeddings).to(self.device), requires_grad=False)

        self.pos_enc_encoder = PositionalEncoding(self.d_model, pos_dropout, max_input_seq_length)
        self.pos_enc_decoder = PositionalEncoding(self.d_model, pos_dropout, max_label_seq_length)

        encoder_norm = nn.LayerNorm(self.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        
        decoder_norm = nn.LayerNorm(self.d_model)
        decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)
    
        self.generator = nn.Linear(self.d_model, vocab_size)
    

    def forward(self, src, tgt, src_attention_mask, tgt_attention_mask, src_key_padding_mask, tgt_key_padding_mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = src.shape[0]
        src = src.to(device)
        tgt = tgt.to(device)
        src_emb = self.embed_src.forward((src))
        tgt_emb = self.embed_tgt.forward((tgt))
        """
        print("src_emb")
        print(src_emb.shape)
        print("tgt_emb")
        print(tgt_emb.shape)
        """
        src_emb = self.sequence_to_first_dimension(src_emb, batch_size)
        tgt_emb = self.sequence_to_first_dimension(tgt_emb, batch_size)
        """
        print("src_emb transposed")
        print(src_emb.shape)
        print("tgt_emb transposed")
        print(tgt_emb.shape)
        """
        src_emb = self.pos_enc_encoder(src_emb)
        tgt_emb = self.pos_enc_decoder(tgt_emb)
        """
        print("src_emb + positional")
        print(src_emb.shape)
        print("tgt_emb + positional")
        print(tgt_emb.shape)
        print()
        print("check masks")
        print("src_key_padding_mask")
        print(src_key_padding_mask.shape)
        print("tgt_attention_mask")
        print(tgt_attention_mask.shape)
        print("src_key_padding_mask")
        print(src_key_padding_mask.shape)
        print("tgt_key_padding_mask")
        print(tgt_key_padding_mask.shape)
        """
        
        #print("encoding...")
        memory = self.transformer_encoder(src_emb, mask=None, src_key_padding_mask=src_key_padding_mask)
        outs = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_attention_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        result = self.generator(outs)
        #print("result")
        #print(result.shape)
        return result

    
    def encode(self, src, src_key_padding_mask, DEVICE):
        batch_size = src.shape[0]
        src_emb = self.embed_src.forward(src)
        src_emb = self.sequence_to_first_dimension(src_emb, batch_size)
        src_emb = self.pos_enc_encoder(src_emb)
        memory = self.transformer_encoder(src_emb, mask=None, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self, tgt, memory, src_key_padding_mask, tgt_attention_mask, tgt_key_padding_mask):
        batch_size = tgt.shape[0] 
        tgt_emb = self.embed_tgt.forward(tgt)
        tgt_emb = self.sequence_to_first_dimension(tgt_emb, batch_size)
        tgt_emb = self.pos_enc_decoder(tgt_emb)
        outs = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_attention_mask, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        result = self.generator(outs)
        #print("result")
        #print(result.shape)
        return result



    def sequence_to_first_dimension(self, tensor, batch_size=None):
        assert batch_size is not None
        assert tensor.shape[0] == batch_size
        return tensor.transpose(0, 1).contiguous()

    def bs_to_first_dimension(self, tensor, batch_size=None):
        assert batch_size is not None
        assert tensor.shape[1] == batch_size
        return tensor.transpose(0, 1).contiguous()




# helper function
def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, DEVICE):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE=DEVICE)
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 0)#.transpose(0, 1)
    tgt_padding_mask = (tgt == 0)#.transpose(0, 1)
    return src_attention_mask, tgt_attention_mask, src_padding_mask, tgt_padding_mask

def create_src_masks(src, DEVICE):
    src_seq_len = src.shape[1]
    src_attention_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_key_padding_mask = (src == 0)
    return src_attention_mask, src_key_padding_mask

def create_tgt_masks(tgt, DEVICE):
    tgt_seq_len = tgt.shape[1]
    tgt_attention_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    tgt_key_padding_mask = (tgt == 0)
    return tgt_attention_mask, tgt_key_padding_mask
