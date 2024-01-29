import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import GPT2LMHeadModel

from bot.config.config import cfg


class CustomDataset(Dataset):

    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        question = sample['Q']
        answer = sample['A']

        encoded_inputs = self.tokenizer(question, answer, padding='max_length', truncation=True, max_length=128,
                                        return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return input_ids, attention_mask,


class LoadDataset:
    def __init__(self, train, token, num_workers=0):
        self.train = train
        self.num_workers = num_workers
        self.tokenizer = token

    @property
    def init_dataset(self):
        train_dataset = CustomDataset(self.train, self.tokenizer)
        return train_dataset

    def init_dataloader(self, train_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=self.num_workers)
        return train_dataloader

    @property
    def load(self):
        train_dataset = self.init_dataset
        train_loader = self.init_dataloader(train_dataset)
        return train_loader



