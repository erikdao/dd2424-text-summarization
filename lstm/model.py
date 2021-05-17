"""
LSTM model
"""
import os

import torch
import torch.nn as nn
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
    def __init__(self, embedding_dim, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

    def forward(self, embedded_text, text_len):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded_text, text_len)
        packed_outputs, hidden = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Decoder(nn.Module):
    """
    The decoder is a single LSTM
    """
    def __init__(self, output_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim):
        self.output_dim = output_dim

        self.lstm = nn.LSTM((enc_hidden_dim * 2) + embedding_dim, dec_hidden_dim)
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embedding_dim, output_dim)

    def forward(self, input_, hidden, encoder_outputs, mask):
        input_ = input_.unsqueeze(0)
        # Read understand it

class LSTMGenerator(pl.LightningModule):
    def __init__(self, input_size=256, hidden_size=256):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size)

        # output
        self.out = nn.Linear(hidden_size, MAX_OUTPUT_LENGTH)
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input, hidden):
        # input = input_batch['input']
        encoded, _ = self.encoder(input)
        output, hidden = self.decoder(encoded)
        print('forward::output', output.size())
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def configure_optimizers(self):
        pass

    def training_step(self, input_batch):
        pass

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)