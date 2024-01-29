from torch.nn.functional import softmax
from tqdm import tqdm


def CalP1(fine_model, dataloader):
    fine_model.eval()
    correct = 0
    for i_batch, data in enumerate(tqdm(dataloader, desc="evaluation")):
        batch_input_tokens, batch_input_attentions, batch_input_labels = data

        batch_input_tokens = batch_input_tokens.cuda()
        batch_input_attentions = batch_input_attentions.cuda()
        batch_input_labels = batch_input_labels.cuda()

        """Prediction"""
        outputs = fine_model(batch_input_tokens, batch_input_attentions)
        probs = softmax(outputs, 1)
        true_probs = probs[:, 1]
        pred_ind = true_probs.argmax(0).item()
        gt_ind = batch_input_labels.argmax(0).item()

        if pred_ind == gt_ind:
            correct += 1
    return round(correct / len(dataloader) * 100, 2)
