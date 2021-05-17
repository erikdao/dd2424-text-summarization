"""
LSTM model
"""
import os
import typing

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from lstm.logger import logger
from utils.glove_embedding import GloveEmbedding

pl.seed_everything(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_OUTPUT_LENGTH = 128


class Encoder(nn.Module):
    """
    The Encoder is a bidirectional LSTM
    """
    def __init__(self,
        embeddings: typing.Optional[typing.Any], embedding_dim: typing.Optional[int], word2index: typing.Optional[typing.Any],
        encoder_hidden_dim: typing.Optional[int], decoder_hidden_dim: typing.Optional[int],
        dropout: typing.Optional[float] = 0.5
    ):

        super().__init__()

        self.embedding = GloveEmbedding(embeddings, embedding_dim, word2index)
        # batch_first = True, as the dim of self.embedding is [batch_size, sentence_length, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, encoder_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

    def forward(self, text):
        """
        Params:
            text - Tensor: [batch_size, sentence_lenghth]

        Returns:
            outputs - Tensor: [batch_size, sentence_length, encoder_hidden_dim]
            hidden - Tensor: [batch_size, decoder_hidden_dim]
        """
        embedded = self.dropout(self.embedding.forward(text))
        outputs, (h_0, c_0) = self.lstm(embedded)

        # Since self.lstm is bidrection, h_0 dim is [2, batch_size, encoder_hidden_dim]
        # and we want to produce a hidden state vector of dim [batch_size, decoder_hidden_dim]
        # Therefore, we concatnate the first dimension of h_0 and feed it to the fc layer
        # of the encoder
        hidden = torch.tanh(self.fc(torch.cat((h_0[-2, :, :], h_0[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    """
    Attention mechanism takes encoder hidden state and decoder hidden state, scores
    them and returns a context vector
    """
    def __init__(self, enc_hid_dim: typing.Optional[int], dec_hid_dim: typing.Optional[int]):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    The Decoder is a single LSTM
    """
    def __init__(self,
        embeddings: typing.Optional[typing.Any],
        word2index: typing.Optional[typing.Any],
        output_dim: typing.Optional[int],
        embedding_dim: typing.Optional[int],
        encoder_hidden_dim: typing.Optional[int],
        decocder_hidden_dim: typing.Optional[int],
        dropout: typing.Optional[float],
        # attention: typing.Optional[Attention],
    ):
        super().__init__()

        self.output_dim = output_dim
        # self.attention = attention

        self.embedding = GloveEmbedding(embeddings, embedding_dim, word2index)
        self.lstm = nn.LSTM((encoder_hidden_dim * 2) + embedding_dim, decocder_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear((encoder_hidden_dim * 2) + decocder_hidden_dim + embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding.forward(input))
        output, (hidden, _) = self.lstm(embedded.unsqueeze(1), (hidden.unsqueeze(0), hidden.unsqueeze(0)))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        pred = self.fc_out(torch.cat((output, embedded), dim=1))
        import pdb; pdb.set_trace();

        return pred, hidden.squeeze(0)


class LSTMSummary(pl.LightningModule):
    """
    Text Summarization model based on bidrectional LSTM encoder-decoder
    """
    def __init__(self,
        vocab_size: typing.Optional[int],
        input_dim: typing.Optional[int],
        output_dim: typing.Optional[int],
        embedding_dim: typing.Optional[int],
        encoder_hidden_dim: typing.Optional[int],
        decoder_hidden_dim: typing.Optional[int],
        encoder_dropout: typing.Optional[float] = 0.5,
        decoder_dropout: typing.Optional[float] = 0.5,
        text_pad_idx: typing.Optional[typing.Any] = None
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim, encoder_hidden_dim, encoder_hidden_dim,
            decoder_hidden_dim, encoder_dropout
        )
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = Decoder(
            output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim,
            decoder_dropout
        )
        self.text_pad_idx = text_pad_idx

        # Initialize weights
        logger.info("Initializing weights for LSTMSummary...")

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            if 'bias' in name:
                torch.nn.init.constant_(param)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)