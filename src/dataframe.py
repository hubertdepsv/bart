# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import datasets


def dataframe_to_pandas(dataset, total_number='all'):
    """
    Transform a transformer dataset into a pandas dataframe

    Args:
        dataset : 
        total_number (optional): limit the number of observation. Defaults to 1000.

    Returns:
        pandas dataframe 
    """
    if total_number=='all':
        total_number = len(dataset)
    else :
        assert total_number <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(total_number):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])

    return df.reset_index(drop=True)


class DataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe # raw dataset
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text] # document
        self.source_text = self.data[source_text] # summary

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze() # from shape [1, source_len] to shape [source_len]
        source_mask = source["attention_mask"].squeeze() # from shape [1, source_len] to shape [source_len]
        target_ids = target["input_ids"].squeeze() # from shape [1, summ_len] to shape [summ_len]
        target_mask = target["attention_mask"].squeeze() # from shape [1, summ_len] to shape [summ_len]

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }