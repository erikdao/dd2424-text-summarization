"""
A custom Pytorch Dataloader for the CSV Review data
"""
import os
import typing
import pandas as pd

import torch
import torch.nn as nn
import torchtext.vocab as vocab
from torchtext.data import get_tokenizer
from torch.utils import data as data_utils

from lstm.config import config
from lstm.glove import extend_glove


class GloveEmbeddingTransform(object):
    """
    This ops transform an input sentence by 1) tokenize the sentence; then 2) build a tensor
    from the embedding tensors of all tokens in the sentence
    """

    def __init__(self):
        self.glove = extend_glove(vocab.GloVe(name="6B", dim=50))
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def _tokenize(self, sentence: typing.Optional[str]) -> typing.List[typing.Any]:
        if type(sentence) != str:
            return []
        sentence = sentence.strip().lower().replace("\n", "")
        punctuation = ["(", ")", ":", '"', " "]
        return [
            tok
            for tok in self.tokenizer(sentence)
            if (not tok.isspace() and tok not in punctuation)
        ]

    def __call__(self, sample):
        input = sample.get("input")
        label = sample.get("label")

        input_tokens = self._tokenize(input)
        if len(input_tokens) > config.INPUT_LENGTH:
            input_tokens = input_tokens[: config.INPUT_LENGTH]
        assert len(input_tokens) <= config.INPUT_LENGTH

        input_vec = [
            self.glove.stoi[token] for token in input_tokens if token in self.glove.stoi
        ]
        while len(input_vec) < config.INPUT_LENGTH:
            input_vec.append(self.glove.stoi[config.PAD_TOKEN])
        assert len(input_vec) == config.INPUT_LENGTH
        input_tensor = torch.tensor(input_vec, dtype=torch.long)

        label_tokens = self._tokenize(label)
        if len(label_tokens) > config.OUTPUT_LENGTH:
            label_tokens = label_tokens[: config.OUTPUT_LENGTH]
        assert len(label_tokens) <= config.INPUT_LENGTH

        label_vec = [
            self.glove.stoi[token] for token in label_tokens if token in self.glove.stoi
        ]
        while len(label_vec) < config.OUTPUT_LENGTH:
            label_vec.append(self.glove.stoi[config.PAD_TOKEN])
        assert len(label_vec) == config.OUTPUT_LENGTH
        label_tensor = torch.tensor(label_vec, dtype=torch.long)

        return {"input": input_tensor, "label": label_tensor}


class ReviewDataset(data_utils.Dataset):
    """
    Class to present the Review Dataset
    """

    def __init__(self, csv_file=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.data.iloc[idx]

        # Construct a sample from the row in the dataframe.the `input` refers to
        # the `text`, while the `label` refers to the `summary` attribute
        sample = {"input": record["text"], "label": record["summary"]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataloader(
    csv_file: typing.Optional[str] = None,
    batch_size: typing.Optional[int] = 1,
    shuffle: typing.Optional[bool] = True,
):
    """
    Create data loader
    """
    dataset = ReviewDataset(csv_file=csv_file, transform=GloveEmbeddingTransform())
    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, drop_last=True
    )

    return loader
