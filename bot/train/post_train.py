from bot.config.config import SaveModel, pcfg
from bot.train.loss import CELoss
from tqdm import tqdm
import torch

def post_train(post_model, post_dataloader, optimizer, scheduler):
    for epoch in range(pcfg.epochs):
        print(f"{epoch} epoch start")
        post_model.train()
        for i_batch, data in enumerate(tqdm(post_dataloader)):
            (
                batch_corrupt_tokens,
                batch_output_tokens,
                batch_corrupt_mask_positions,
                batch_urc_inputs,
                batch_urc_labels,
                batch_mlm_attentions,
                batch_urc_attentions,
            ) = data
            batch_corrupt_tokens = batch_corrupt_tokens.cuda()
            batch_output_tokens = batch_output_tokens.cuda()
            batch_urc_inputs = batch_urc_inputs.cuda()
            batch_urc_labels = batch_urc_labels.cuda()
            batch_mlm_attentions = batch_mlm_attentions.cuda()
            batch_urc_attentions = batch_urc_attentions.cuda()

            """Prediction"""
            corrupt_mask_outputs, urc_cls_outputs = post_model(
                batch_corrupt_tokens,
                batch_corrupt_mask_positions,
                batch_urc_inputs,
                batch_mlm_attentions,
                batch_urc_attentions,
            )

            """Loss calculation & training"""
            original_token_indexs = []
            for i_batch in range(len(batch_corrupt_mask_positions)):
                original_token_index = []
                batch_corrupt_mask_position = batch_corrupt_mask_positions[i_batch]
                for pos in batch_corrupt_mask_position:
                    original_token_index.append(batch_output_tokens[i_batch, pos].item())
                original_token_indexs.append(original_token_index)

            mlm_loss = 0
            for corrupt_mask_output, original_token_index in zip(
                corrupt_mask_outputs, original_token_indexs
            ):
                mlm_loss += CELoss(
                    corrupt_mask_output, torch.tensor(original_token_index).cuda()
                )
            urc_loss = CELoss(urc_cls_outputs, batch_urc_labels)

            loss_val = mlm_loss + urc_loss

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(post_model.parameters(), pcfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    SaveModel(post_model, ".")