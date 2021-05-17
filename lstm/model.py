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

pl.seed_everything(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_OUTPUT_LENGTH = 128


class Encoder(nn.Module):
    """
    The Encoder is a bidirectional LSTM
    """
    def __init__(self,
        vocab_size: typing.Optional[int], embedding_dim: typing.Optional[int],
        encoder_hidden_dim: typing.Optional[int], decoder_hidden_dim: typing.Optional[int],
        dropout: typing.Optional[float] = 0.1
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

        # # Initialize weights
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         torch.nn

    def forward(self, text, text_len):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

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
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, hidden, encoder_outputs, mask):
        input_ = input_.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_))

        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)
        rnn_input_ = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input_, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0), attn.squeeze(1)


class LSTMSummary(pl.LightningModule):
    """
    Text Summarization model based on bidrectional LSTM encoder-decoder
    """
    def __init__(self,
        vocab_size: typing.Optional[int], embedding_dim: typing.Optional[int],
        encoder_hidden_dim: typing.Optional[int], decoder_hidden_dim: typing.Optional[int],
        encoder_dropout: typing.Optional[float] = 0.5):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = decoder
        self.text_pad_idx = text_pad_idx
    
    # def forward(self, input_batch):
    #     batch_size = input_batch.shape[1]
    #     summary_vocab_size = self.decoder.output_dim

    #     outputs = torch.zeros(256, batch_size, summary_vocab_size)
    #     encoder_outputs, hidden = self.encoder(input_batch, )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)