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
        
        self.pos_enc_encoder = PositionalEncoding(self.d_model, pos_dropout, max_input_seq_length)
        self.pos_enc_decoder = PositionalEncoding(self.d_model, pos_dropout, max_label_seq_length)

        encoder_norm = nn.LayerNorm(self.d_model).to(self.device)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        
        decoder_norm = nn.LayerNorm(self.d_model).to(self.device)
        decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)
    
        self.generator = nn.Linear(self.d_model, vocab_size)
    

    def forward(self, src, tgt, src_attention_mask=None, tgt_attention_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = src.shape[0]
        src = src.to(device)
        tgt = tgt.to(device)
        src_emb = self.embed_src.forward((src))
        tgt_emb = self.embed_tgt.forward((tgt))
        src_emb = self.pos_enc_encoder(src_emb)
        tgt_emb = self.pos_enc_decoder(tgt_emb)
        
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





