import typing

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class Encoder(pl.LightningModule):
    """
    The Encoder is a LSTM
    """
    def __init__(self, input_size: typing.Optional[int], hidden_size: typing.Optional[int]):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # .view(1, 1, -1)
        output = embedded.unsqueeze(0)
        output, hidden = self.lstm(output, hidden)
        return output, hidden
    

class Decoder(pl.LightningModule):
    def __init__(self, hidden_size: typing.Optional[int], output_size: typing.Optional[int]):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input, hidden):
        output = self.embedding(input)  # .view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output.unsqueeze(0), hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class LSTMSummary(pl.LightningModule):
    def __init__(self, input_size: typing.Optional[int],
                       hidden_size: typing.Optional[int],
                       output_size: typing.Optional[int]):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

        # Initialize weights
        for name, params in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(params, 0.0)
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(params)
    
    def forward(self, input_seq):
        """
        Params:
            input_seq - Tensor: [batch_size, sentence_length]
        """
        input_length = input_seq.shape[1]

        # Encode the input
        encoder_outputs = torch.zeros(128, self.encoder.hidden_size)
        encoder_h0 = torch.zeros(1, 1, self.encoder.hidden_size)
        encoder_c0 = torch.zeros(1, 1, self.encoder.hidden_size)
        encoder_hidden = (encoder_h0, encoder_c0)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_seq[:, ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # Decode hidden state
        SOS_Token, EOS_Token = 0, 1
        decoder_input = torch.tensor([[SOS_Token]])
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(64)

        for di in range(64):
            decoder_ouput, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_ouput.topk(1)
            decoder_input = topi.squeeze().detach()
            decoder_outputs[di] = decoder_input
            if decoder_input.item() == EOS_Token:
                break
        
        return decoder_outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
