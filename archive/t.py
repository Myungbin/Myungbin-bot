import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model  # 텍스트


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['Q']
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['A']  # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            return {
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'question': question['input_ids'].squeeze(),
            }


train = pd.read_csv(r"C:\MB_Project\project\Myungbin-bot\data\processed\QnA_data.csv")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)
train_dataset = VQADataset(train, tokenizer, is_test=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")
model.resize_token_embeddings(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
device = 'cuda'


def train(model, loader, optimizer, criterion):
    model.to(device)
    model.train()
    total_loss = 0

    for data in tqdm(loader, total=len(loader)):
        inputs = data[:, :-1].to(device)  # 마지막 토큰은 정답이 아닌 입력으로 사용
        labels = data[:, 1:].to(device)  # 다음 토큰을 정답으로 사용

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader)
    return avg_loss


for epoch in range(1):
    avg_loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")
