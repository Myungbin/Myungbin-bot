import pickle
import torch
from collections import OrderedDict
from tqdm import tqdm
from bot.models.fine_model import FineModel
import torch.nn as nn


def load_candidates(file_path):
    """Load candidates from pickle file and remove duplicates"""
    with open(file_path, "rb") as file:
        candidates = pickle.load(file)
    return list(OrderedDict.fromkeys(candidates))


def tokenize_context(fine_model, context):
    context_token = [fine_model.tokenizer.cls_token_id]
    for utt in context:
        context_token += fine_model.tokenizer.encode(utt, add_special_tokens=False)
        context_token += [fine_model.tokenizer.sep_token_id]
    return context_token


def evaluate_candidates(fine_model, context_token, candidates, batch_size=32):
    batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]
    candidate_results = []
    for batch in tqdm(batches):
        session_tokens = []
        for response in batch:
            response_token = [fine_model.tokenizer.eos_token_id]
            response_token += fine_model.tokenizer.encode(
                response, add_special_tokens=False, truncation=True, max_length=64
            )
            candidate_tokens = context_token + response_token
            session_tokens.append(candidate_tokens)
        batch_input_tokens, batch_input_attentions = prepare_batch_input(session_tokens, fine_model)
        candidate_results.extend(get_batch_results(fine_model, batch, batch_input_tokens, batch_input_attentions))
    return candidate_results


def prepare_batch_input(session_tokens, fine_model):
    max_input_len = max(len(session_token) for session_token in session_tokens)
    batch_input_tokens = []
    batch_input_attentions = []
    for session_token in session_tokens:
        input_token = session_token + [fine_model.tokenizer.pad_token_id] * (max_input_len - len(session_token))
        input_attention = [1] * len(session_token) + [0] * (max_input_len - len(session_token))
        batch_input_tokens.append(input_token)
        batch_input_attentions.append(input_attention)
    return torch.tensor(batch_input_tokens).cuda(), torch.tensor(batch_input_attentions).cuda()


def get_batch_results(fine_model, batch, batch_input_tokens, batch_input_attentions):
    softmax = nn.Softmax(dim=1)
    results = fine_model(batch_input_tokens, batch_input_attentions)
    prob = softmax(results)
    true_prob = prob[:, 1].tolist()
    return [(utt, prob) for utt, prob in zip(batch, true_prob)]


def select_top_candidates(candidate_results, top_n=10):
    candidate_results.sort(key=lambda x: x[1], reverse=True)
    return candidate_results[:top_n]


if __name__ == "__main__":
    # 메인 코드
    pickle_file_path = "candidates_new1.pkl"
    fine_model = FineModel().to("cuda")
    fine_model.load_state_dict(torch.load("fine_model.pth"), strict=False)
    context = ["점메추"]

    candidates = load_candidates(pickle_file_path)
    context_token = tokenize_context(fine_model, context)
    candidate_results = evaluate_candidates(fine_model, context_token, candidates)
    top_candidates = select_top_candidates(candidate_results, 100)

    # 선택된 상위 후보 출력
    print(context)
    for utt, prob in top_candidates:
        print(utt, "##", round(prob, 3))
