import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from bot.config.config import pcfg
from bot.data.post_loader import PostLoader
from bot.models.post_model import PostModel
from bot.train.post_train import post_train

post_dataset = PostLoader(pcfg.data_path)
post_dataloader = DataLoader(
    post_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=post_dataset.collate_fn,
)
post_model = PostModel().cuda()

num_training_steps = len(post_dataset) * pcfg.epochs
num_warmup_steps = len(post_dataset)

optimizer = torch.optim.AdamW(post_model.parameters(), lr=pcfg.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)
post_train(post_model, post_dataloader, optimizer, scheduler)
