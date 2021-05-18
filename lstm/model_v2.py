from typing import Any, Optional, List
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from config import config


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class LSTMSummary(pl.LightningModule):
    def __init__(self, input_size: Optional[int], hidden_size: Optional[int], output_size: Optional[int]):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

        # Initialize weights:
        for name, params in self.named_parameters():
            if 'embedding' in name:
                # Don't initialize the weights of the embedding layer here
                # as they'll be initialized from the pretrained glove
                continue
            if 'bias' in name:
                torch.nn.init.constant_(params, 0.0)
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(params)
    
    def forward(self, data):
        input_tensor = data['input']
        label_tensor = data['label']

        input_length = input_tensor.shape[1]
        target_length = label_tensor.shape[1]
        print("input_tensor", input_tensor.size(), "label_tensor", label_tensor.size())

        # Encode phase
        encoder_outputs = torch.zeros(config.INPUT_LENGTH, self.encoder.hidden_size)
        encoder_hidden = self.encoder.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[:, ei], encoder_hidden)
            # print(ei, 'encoder_ouput', encoder_output.size())
            # print(ei, 'encoder_outputs', encoder_outputs.size())
            encoder_outputs[ei] = encoder_output[0, 0]
        
        # Decode phase
        decoder_hidden = encoder_hidden
        decoder_input = label_tensor[0, :]

        for di in range(target_length):
            import pdb; pdb.set_trace();
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            print(di, decoder_input.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)