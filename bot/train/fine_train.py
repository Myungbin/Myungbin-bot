import torch
from tqdm import tqdm

from bot.config.config import FineSaveModel as SaveModel
from bot.config.config import fcfg as cfg
from bot.train.loss import CELoss
from bot.train.metric import CalP1


def fine_train(fine_model, train_dataloader, dev_dataloader, optimizer, scheduler):
    best_p1 = 0
    for epoch in range(cfg.epochs):
        fine_model.train()
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            batch_input_tokens, batch_input_attentions, batch_input_labels = data

            batch_input_tokens = batch_input_tokens.cuda()
            batch_input_attentions = batch_input_attentions.cuda()
            batch_input_labels = batch_input_labels.cuda()

            """Prediction"""
            outputs = fine_model(batch_input_tokens, batch_input_attentions)
            loss_val = CELoss(outputs, batch_input_labels)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(fine_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        fine_model.eval()
        p1 = CalP1(fine_model, dev_dataloader)
        print(f"Epoch: {epoch}번째 모델 성능(p@1): {p1}")
        if p1 > best_p1:
            best_p1 = p1
            model_name = f"fine_model_{epoch}.pth"
            print(f"BEST 성능(p@1): {best_p1}")
            SaveModel(fine_model, ".", model_name)
