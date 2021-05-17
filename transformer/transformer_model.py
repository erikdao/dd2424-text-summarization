import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import positional_encoding

from utils/glove_embedding import *

class SummaryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(vocab_size, d_model) # TODO check to use GLOVE
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        # glove
        glove = GloveEmbedding()
        self.word_embedding = glove.word_embedding

        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fully_connected = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        src = rearrange(src, 'n s -> s n')
        tgt = rearrange(tgt, 'n t -> t n')
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = rearrange(output, 't n e -> n t e')
        return self.fully_connected(output)


def main():
    transformer_model = SummaryTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout)
    
    # TODO DATA tokinization and masking (Adam)
    """
    train_loader = create_dataloader(
        csv_file='data/amazon-product-reviews/Reviews.csv',
        shuffle=True
    )
    for idx, data in enumerate(train_loader):
        if idx == 2:
            input_vec = data['input']
            label_vec = data['label']
            print(input_vec, input_vec.shape)
            print(label_vec, label_vec.shape)
            break
    """
if __name__ == '__main__':
    main()
