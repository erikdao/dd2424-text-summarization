"""
For quick reference
    nn.Embedding - input [seq_length, batch_size]   output [seq_length, batch_size, embedding_size]
    nn.LSTM      - input []
"""
from typing import Any, Optional
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from lstm.config import config
from lstm.logger import logger


class Encoder(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: Optional[int] = 1,
        word2index: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = input_size
        self.num_layers = num_layers  # number of LSTM layers

        self.embedding = nn.Embedding(
            input_size, config.EMBEDDING_DIM, padding_idx=word2index[config.PAD_TOKEN]
        )
        self.lstm = nn.LSTM(config.EMBEDDING_DIM, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # .view(1, 1, -1)
        output, hidden_cell = self.lstm(embedded, hidden)
        # h_0 = torch.tanh(hidden_cell[0])
        # c_0 = torch.tanh(hidden_cell[1])
        # output = torch.tanh(output)
        return output, hidden_cell  # (h_0, c_0)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, config.BATCH_SIZE, self.hidden_size).to(
            self.device
        )
        cell = torch.zeros(self.num_layers, config.BATCH_SIZE, self.hidden_size).to(
            self.device
        )
        return (hidden, cell)


class Decoder(pl.LightningModule):
    def __init__(
        self,
        hidden_size,
        output_size,
        num_layers: Optional[int] = 1,
        word2index: Optional[Any] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            output_size, config.EMBEDDING_DIM, padding_idx=word2index[config.PAD_TOKEN]
        )
        self.lstm = nn.LSTM(config.EMBEDDING_DIM, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        # output = torch.tanh(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, config.EMBEDDING_DIM, self.hidden_size).to(
            self.device
        )


class LSTMSummary(pl.LightningModule):
    def __init__(
        self,
        embedding_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        word2index: Optional[Any] = None,
        index2word: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.word2index = word2index
        self.index2word = index2word
        self.embeddings = embeddings

        self.encoder = Encoder(embedding_size, hidden_size, word2index=word2index)
        self.decoder = Decoder(hidden_size, embedding_size, word2index=word2index)

        # Initialize weights:
        for name, params in self.named_parameters():
            if "embedding" in name:
                # Don't initialize the weights of the embedding layer here
                # as they'll be initialized from the pretrained glove
                params.requires_grad = False
                params.data.copy_(torch.from_numpy(self.embeddings))
            if "bias" in name:
                torch.nn.init.constant_(params, 0.0)
            if "weight" in name:
                torch.nn.init.xavier_uniform_(params)

    def forward(self, input_tensor):
        input_length = input_tensor.shape[0]

        # Encode phase
        encoder_hidden = self.encoder.init_hidden()

        for ei in range(input_length):
            # Note, we do input_tensor[ei, :].unsequeeze(0) below to make the input to
            # dim [1, batch_size] (i.e., batches of 1 token) to fit nn.Embedding requirements
            _, encoder_hidden = self.encoder(
                input_tensor[ei, :].unsqueeze(0), encoder_hidden
            )

        # Decode phase
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(
            [[self.word2index[config.SOS_TOKEN]] * config.BATCH_SIZE]
        ).to(self.device)
        outputs = torch.full(
            (config.OUTPUT_LENGTH, config.BATCH_SIZE, self.embedding_size),
            self.word2index[config.PAD_TOKEN],
            dtype=torch.float,
        ).to(self.device)

        output_sentence = []
        for di in range(config.OUTPUT_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[di] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.permute(-1, 0).detach()
            output_sentence.append(decoder_input)

        # import pdb; pdb.set_trace()
        if self.global_step % config.PRINT_PREDICTION_STEP == 0:
            pred = " ".join(
                [
                    self.index2word[di[:, 0].item()]
                    for di in output_sentence
                    if di[:, 0].item() in self.index2word.keys()
                    and self.index2word[di[:, 0].item()] != config.PAD_TOKEN
                ]
            )
            logger.debug(f"Prediction: {pred}")

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad == True],
            lr=config.LEARNING_RATE,
        )

    # def configure_optimizers(self):
    #     return torch.optim.SGD([param for param in self.parameters() if param.requires_grad == True], lr=0.1, momentum=0.9)

    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        input_tensor = train_batch["input"]
        label_tensor = train_batch["label"]
        # input_tensor is of dim [batch_size, seq_len], but nn.Embedding accepts
        # [seq_len, batch_size] so we have to switch the axies
        pred_tensor = self.forward(input_tensor.permute(1, 0))
        loss = F.cross_entropy(
            pred_tensor.float().view(-1, self.embedding_size), label_tensor.view(-1)
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, val_idx) -> Optional[STEP_OUTPUT]:
        input_tensor = val_batch["input"]
        label_tensor = val_batch["label"]
        # input_tensor is of dim [batch_size, seq_len], but nn.Embedding accepts
        # [seq_len, batch_size] so we have to switch the axies
        pred_tensor = self.forward(input_tensor.permute(1, 0))
        # At this step, pred_tensor should be of dim [seq_len, batch_size, embedding_size]
        loss = F.cross_entropy(
            pred_tensor.float().view(-1, self.embedding_size), label_tensor.view(-1)
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
