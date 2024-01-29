import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from bot.config.config import fcfg as cfg
from bot.data.fine_loader import FineLoader
from bot.models.fine_model import FineModel
from bot.train.fine_train import fine_train

train_path = "data/processed/train.json"
train_dataset = FineLoader(train_path)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=train_dataset.collate_fn,
)

dev_path = "data/processed/dev.json"
dev_dataset = FineLoader(dev_path)
dev_dataloader = DataLoader(
    dev_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=dev_dataset.collate_fn,
)

fine_model = FineModel().cuda()
fine_model.load_state_dict(torch.load("post_model.pt"), strict=False)
num_training_steps = len(train_dataset) * cfg.epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(fine_model.parameters(), lr=cfg.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

fine_train(fine_model, train_dataloader, dev_dataloader, optimizer, scheduler)
