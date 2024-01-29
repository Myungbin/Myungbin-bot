import torch


def test(model, tokenizer, input_sentence):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        attention_mask = (input_ids != tokenizer.pad_token_id).float()

        output = model(input_ids, attention_mask)
        _, predicted_ids = torch.max(output, dim=-1)

        predicted_sentence = tokenizer.decode(predicted_ids.squeeze().tolist(), skip_special_tokens=True)
        return predicted_sentence
