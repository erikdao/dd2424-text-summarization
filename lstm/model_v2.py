from typing import Any, Optional, List
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.optim
import torch.nn as nn
import torchtext.vocab as vocab
import torch.nn.functional as F

import pytorch_lightning as pl

from config import config
from glove import extend_glove

glove = extend_glove(vocab.GloVe(name="6B", dim=50))


class Encoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = input_size

        self.embedding = nn.Embedding(
            input_size, 50, padding_idx=glove.stoi[config.PAD_TOKEN]
        )
        self.lstm = nn.LSTM(50, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(1, 1, self.hidden_size).to(self.device)
        cell = torch.zeros(1, 1, self.hidden_size).to(self.device)
        return (hidden, cell)


class Decoder(pl.LightningModule):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            output_size, 50, padding_idx=glove.stoi[config.PAD_TOKEN]
        )
        self.lstm = nn.LSTM(50, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)  # .unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(self.device)


class LSTMSummary(pl.LightningModule):
    def __init__(
        self,
        input_size: Optional[int],
        hidden_size: Optional[int],
        output_size: Optional[int],
    ):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

        # Initialize weights:
        for name, params in self.named_parameters():
            if "embedding" in name:
                params.requires_grad = False
                params.data.copy_(glove.vectors)
                # Don't initialize the weights of the embedding layer here
                # as they'll be initialized from the pretrained glove
                # continue
            if "bias" in name:
                torch.nn.init.constant_(params, 0.0)
            if "weight" in name:
                torch.nn.init.xavier_uniform_(params)

    def forward(self, input_tensor):
        input_length = input_tensor.shape[1]

        # Encode phase
        encoder_outputs = torch.zeros(
            config.INPUT_LENGTH, self.encoder.hidden_size, requires_grad=True
        ).to(self.device)
        encoder_hidden = self.encoder.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[:, ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        # Decode phase
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([glove.stoi[config.PAD_TOKEN]]).to(
            self.device
        )  # .unsqueeze(0)

        outputs = torch.full(
            (config.OUTPUT_LENGTH, self.encoder.vocab_size),
            glove.stoi[config.PAD_TOKEN],
            dtype=torch.float,
            requires_grad=True,
        ).to(self.device)

        output_sentence = []
        for di in range(config.OUTPUT_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[di] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            output_sentence.append(decoder_input)
            # if (decoder_input == glove.stoi[config.PAD_TOKEN]):
            #     break

        if self.global_step % 200 == 0:
            pred = " ".join(
                [
                    glove.itos[di.item()]
                    for di in output_sentence
                    if di != glove.stoi[config.PAD_TOKEN]
                ]
            )
            print("Prediction\t", pred)

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad == True],
            lr=config.LEARNING_RATE,
        )

    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        input_tensor = train_batch["input"]
        label_tensor = train_batch["label"]

        pred_tensor = self.forward(input_tensor)
        loss = F.cross_entropy(pred_tensor.float(), label_tensor.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, val_idx) -> Optional[STEP_OUTPUT]:
        input_tensor = val_batch["input"]
        label_tensor = val_batch["label"]

        pred_tensor = self.forward(input_tensor)
        loss = F.cross_entropy(pred_tensor.float(), label_tensor.squeeze())
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
