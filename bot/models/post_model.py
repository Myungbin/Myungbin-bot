import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForMaskedLM


class PostModel(nn.Module):
    def __init__(self):
        super(PostModel, self).__init__()
        self.model = RobertaForMaskedLM.from_pretrained("klue/roberta-base", max_length=512)
        self.hiddenDim = self.model.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", truncation=True, max_length=512)
        special_tokens = {"sep_token": "<SEP>"}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        """ score matrix """
        self.W = nn.Linear(self.hiddenDim, 3)

    def forward(
        self,
        batch_corrupt_tokens,
        batch_corrupt_mask_positions,
        batch_urc_inputs,
        batch_mlm_attentions,
        batch_urc_attentions,
    ):
        corrupt_outputs = self.model(batch_corrupt_tokens, attention_mask=batch_mlm_attentions)["logits"]
        corrupt_mask_outputs = []
        for i_batch in range(len(batch_corrupt_mask_positions)):
            corrupt_mask_output = []
            batch_corrupt_mask_position = batch_corrupt_mask_positions[i_batch]
            if len(batch_corrupt_mask_position) > 0:
                for pos in batch_corrupt_mask_position:
                    corrupt_mask_output.append(corrupt_outputs[i_batch, pos, :].unsqueeze(0))
                corrupt_mask_outputs.append(torch.cat(corrupt_mask_output, 0))
            else:
                corrupt_mask_outputs.append(torch.zeros(1, corrupt_outputs.shape[-1]))

        # URC
        urc_outputs = self.model(batch_urc_inputs, attention_mask=batch_urc_attentions, output_hidden_states=True)[
            "hidden_states"
        ][-1]
        urc_logits = self.W(urc_outputs)
        urc_cls_outputs = urc_logits[:, 0, :]

        return corrupt_mask_outputs, urc_cls_outputs
