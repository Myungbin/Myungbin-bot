import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from bot.models.fine_model import FineModel

pickle_file_path = "candidates.pkl"


fine_model = FineModel().to("cuda")
fine_model.load_state_dict(torch.load("fine_model.pth"), strict=False)
context = ["너는 취미가 뭐니", "취미? 없는데 이제 만들러 가야지"]
with open(pickle_file_path, "rb") as file:
    candidates = pickle.load(file)


context_token = [fine_model.tokenizer.cls_token_id]
for utt in context:
    context_token += fine_model.tokenizer.encode(utt, add_special_tokens=False)
    context_token += [fine_model.tokenizer.sep_token_id]

print(len(context_token))
session_tokens = []
for response in tqdm(candidates):
    response_token = [fine_model.tokenizer.eos_token_id]
    response_token += fine_model.tokenizer.encode(response,
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=fine_model.tokenizer.model_max_length - len(context_token) - 1)
    candidate_tokens = context_token + response_token
    session_tokens.append(candidate_tokens)

# 최대 길이 찾기 for padding
max_input_len = 0
input_tokens_len = [len(x) for x in session_tokens]
max_input_len = max(max_input_len, max(input_tokens_len))

batch_input_tokens = []
batch_input_attentions = []
for session_token in tqdm(session_tokens):
    input_token = session_token + [
        fine_model.tokenizer.pad_token_id
        for _ in range(max_input_len - len(session_token))
    ]
    input_attention = [1 for _ in range(len(session_token))] + [
        0 for _ in range(max_input_len - len(session_token))
    ]
    batch_input_tokens.append(input_token)
    batch_input_attentions.append(input_attention)

batch_input_tokens = torch.tensor(batch_input_tokens).cuda()
batch_input_attentions = torch.tensor(batch_input_attentions).cuda()


softmax = nn.Softmax(dim=1)
results = fine_model(batch_input_tokens, batch_input_attentions)
prob = softmax(results)
true_prob = prob[:, 1].tolist()


print(context)
candidate_results = []
for utt, prob in zip(candidates, true_prob):
    candidate_results.append((utt, prob))

# 확률에 따라 후보 정렬
candidate_results.sort(key=lambda x: x[1], reverse=True)

# 상위 10개 후보 선택
top_10_candidates = candidate_results[:10]

# 선택된 상위 10개 후보 출력
for utt, prob in top_10_candidates:
    print(utt, "##", round(prob, 3))
