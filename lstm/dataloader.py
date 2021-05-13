"""
A custom Pytorch Dataloader for the CSV Review data
"""
import os
import pandas as pd

import torch
from torch.utils import data as data_utils


class ReviewDataset(data_utils.Dataset):
    """
    Class to present the Review Dataset
    """
    def __init__(self, csv_file=None):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        record = self.data.iloc[idx]

        # Construct a sample from the row in the dataframe.the `input` refers to
        # the `text`, while the `label` refers to the `summary` attribute
        sample = {'input': record['Text'], 'label': record['Summary']}

        return sample
