"""
A custom Pytorch Dataloader for the CSV Review data
"""
import os
import typing
import pandas as pd

import torch
import torch.nn as nn
from torch.utils import data as data_utils
from torchvision import transforms, utils
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel

# Max length of sequence, this is a setting for encode_plus of DistilBertTokenizer
# (or more specifically, transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus)
MAX_SEQ_LENGTH = 256


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(x: str) -> typing.Tuple[typing.Any, typing.Any]:
    """
    Tokenize the input string using DistillBertTokenize
    Returns:
        - input_ids: List of token ids to be fed to a model.
        - input_id_type: which we won't care about
    """
    encoding = tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()


class DistilBertSequence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
    def forward(self, input_ids=None, attention_mask=None): #,head_mask=None,labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask)
        hidden_state = distilbert_output[0]                    
        pooled_output = hidden_state[:, 0]  
        return pooled_output

distillBertSequenceModel = DistilBertSequence(DistilBertConfig())


class DistillBERTEncodeTransform(object):
    """
    This transform ops convert the original string, into a 
    word vector using BERT embedding
    """
    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        input_ids, input_attn_mask = tokenize(input)
        label_ids, label_attn_mask = tokenize(label)

        with torch.no_grad():
            x = input_ids.reshape(1, len(input_ids))
            y = label_ids.reshape(1, len(label_ids))

            input_embedded = distillBertSequenceModel.forward(input_ids=x, attention_mask=input_attn_mask)
            label_embedded = distillBertSequenceModel.forward(input_ids=y, attention_mask=label_attn_mask)

            return {'input': input_embedded, 'label': label_embedded}


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
        sample = {'input': record['Text'], 'label': record['Summary']}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataloader(
    csv_file: typing.Optional[str] = None,
    batch_size: typing.Optional[int] = 16,
    shuffle: typing.Optional[bool] = True
):
    """
    Create data loader
    """
    dataset = ReviewDataset(csv_file=csv_file, transform=DistillBERTEncodeTransform())
    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return loader
