import random
import re

import torch
from torch.utils.data import  Dataset
from transformers import AutoTokenizer

class PostLoader(Dataset):
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "klue/roberta-base", turncation=True, padding="max_length", max_length=512
        )
        special_tokens = {"sep_token": "<SEP>"}
        self.tokenizer.add_special_tokens(special_tokens)

        self.data_path = data_path

        conversations = self.load_and_process_data(self.data_path)
        session_dataset = self.merge_and_trim_messages(conversations)

        """ short session context """
        k = 4  # 논문에서 가장 좋았던 숫자
        self.short_session_dataset = []
        for session in session_dataset:
            for i in range(len(session) - k + 1):
                self.short_session_dataset.append(session[i : i + k])

        """ 모든 발화 저장 """
        self.all_utts = set()
        for session in session_dataset:
            for utt in session:
                self.all_utts.add(utt)
        self.all_utts = list(self.all_utts)

    @staticmethod
    def load_and_process_data(file_path):
        # 파일 읽기
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 대화 데이터 추출을 위한 정규 표현식
        pattern = r"\[(.*?)\] \[(.*?)\] (.*)"

        # 대화 데이터 저장을 위한 리스트
        conversations = []

        # 파일의 각 줄을 순회하며 대화 데이터 추출
        for line in lines:
            match = re.match(pattern, line)
            if match:
                speaker = match.group(1)  # 발화자
                time = match.group(2)  # 시간
                message = match.group(3)  # 메시지
                conversations.append((speaker, time, message))

        return conversations

    @staticmethod
    def merge_and_trim_messages(conversations, min_length=5, max_total_length=50):
        reformatted_data = []
        conversation = []
        current_message = ""

        for i in range(len(conversations)):
            _, _, message = conversations[i]

            # 현재 메시지에 추가
            if current_message:
                current_message += " " + message
            else:
                current_message = message

            # 메시지가 최소 단어 길이에 도달하거나 대화가 종료되면 새로운 메시지 시작
            if (
                len(current_message.split()) >= min_length
                or i == len(conversations) - 1
            ):
                # 전체 글자 수가 최대 길이를 초과하면 줄임
                if len(current_message) > max_total_length:
                    words = current_message.split()
                    trimmed_message = " ".join(words[:min_length])  # 최소 단어 길이만큼만 유지
                    conversation.append(trimmed_message)
                    current_message = " ".join(words[min_length:])  # 남은 부분은 다음 메시지로 이동
                else:
                    conversation.append(current_message)
                    current_message = ""

            if len(conversation) == 10:  # 10개의 메시지를 모으면 대화 추가
                reformatted_data.append(conversation)
                conversation = []

        # 마지막 대화 추가
        if conversation:
            reformatted_data.append(conversation)

        return reformatted_data

    @staticmethod
    def merge_short_messages(conversations, min_length=5):
        reformatted_data = []
        conversation = []
        current_message = ""

        for i in range(len(conversations)):
            _, _, message = conversations[i]

            # 현재 메시지에 추가
            if current_message:
                current_message += " "
            current_message += message

            # 메시지가 최소 길이에 도달하거나 대화가 종료되면 새로운 메시지 시작
            if (
                len(current_message.split()) >= min_length
                or i == len(conversations) - 1
            ):
                conversation.append(current_message)
                current_message = ""

            if len(conversation) == 10:  # 10개의 메시지를 모으면 대화 추가
                reformatted_data.append(conversation)
                conversation = []

        # 마지막 대화 추가
        if conversation:
            reformatted_data.append(conversation)

        return reformatted_data

    def __len__(self):  # 기본적인 구성
        return len(self.short_session_dataset)

    def __getitem__(self, idx):  # 기본적인 구성
        session = self.short_session_dataset[idx]
        """ MLM 입력 """
        mask_ratio = 0.15
        self.corrupt_tokens = []
        self.output_tokens = []
        for i, utt in enumerate(session):
            original_token = self.tokenizer.encode(
                utt, add_special_tokens=False, max_length=512, truncation=True
            )

            mask_num = int(len(original_token) * mask_ratio)
            mask_positions = random.sample(
                [x for x in range(len(original_token))], mask_num
            )
            corrupt_token = []
            for pos in range(len(original_token)):
                if pos in mask_positions:
                    corrupt_token.append(self.tokenizer.mask_token_id)
                else:
                    corrupt_token.append(original_token[pos])

            if i == len(session) - 1:
                self.output_tokens += original_token
                self.corrupt_tokens += corrupt_token
            else:
                self.output_tokens += original_token + [self.tokenizer.sep_token_id]
                self.corrupt_tokens += corrupt_token + [self.tokenizer.sep_token_id]

        """ label for loss """
        self.corrupt_mask_positions = []
        for pos in range(len(self.corrupt_tokens)):
            if self.corrupt_tokens[pos] == self.tokenizer.mask_token_id:
                self.corrupt_mask_positions.append(pos)

        """ URC 입력 """
        urc_tokens = []
        context_utts = []
        for i in range(len(session)):
            utt = session[i]
            original_token = self.tokenizer.encode(
                utt, add_special_tokens=False, max_length=512, truncation=True
            )
            if i == len(session) - 1:
                urc_tokens += [self.tokenizer.eos_token_id]
                """ 기존 response 입력 """
                self.positive_tokens = (
                    [self.tokenizer.cls_token_id] + urc_tokens + original_token
                )
                """ random negative respons 입력 """
                while True:
                    random_neg_response = random.choice(self.all_utts)
                    if random_neg_response not in context_utts:
                        break
                random_neg_response_token = self.tokenizer.encode(
                    random_neg_response,
                    add_special_tokens=False,
                    max_length=512,
                    truncation=True,
                )
                self.random_tokens = (
                    [self.tokenizer.cls_token_id]
                    + urc_tokens
                    + random_neg_response_token
                )
                """ context negative response 입력 """
                context_neg_response = random.choice(context_utts)
                context_neg_response_token = self.tokenizer.encode(
                    context_neg_response,
                    add_special_tokens=False,
                    max_length=512,
                    truncation=True,
                )
                self.context_neg_tokens = (
                    [self.tokenizer.cls_token_id]
                    + urc_tokens
                    + context_neg_response_token
                )
            else:
                urc_tokens += original_token + [self.tokenizer.sep_token_id]
            context_utts.append(utt)

        return (
            self.corrupt_tokens,
            self.output_tokens,
            self.corrupt_mask_positions,
            [self.positive_tokens, self.random_tokens, self.context_neg_tokens],
            [0, 1, 2],
        )

    def collate_fn(self, sessions):
        """
        input:
            data: [(session1), (session2), ... ]
        return:
            batch_corrupt_tokens: (B, L) padded (limited to max_length)
            batch_output_tokens: (B, L) padded (limited to max_length)
            batch_corrupt_mask_positions: list
            batch_urc_inputs: (B, L) padded (limited to max_length)
            batch_urc_labels: (B)
            batch_mlm_attentions
            batch_urc_attentions

        batch size: B
        MLM = 3개의 입력데이터 (입력데이터별로 길이가 다름)
        URC = 9개의 입력데이터 (context는 길이가 다름, response candidate도 길이가 다름)
        """
        max_length = 50  # 최대 길이를 512로 설정

        (
            batch_corrupt_tokens,
            batch_output_tokens,
            batch_corrupt_mask_positions,
            batch_urc_inputs,
            batch_urc_labels,
        ) = ([], [], [], [], [])
        batch_mlm_attentions, batch_urc_attentions = [], []

        for session in sessions:
            (
                corrupt_tokens,
                output_tokens,
                corrupt_mask_positions,
                urc_inputs,
                urc_labels,
            ) = session

            # MLM, URC 입력에 대해서 최대 길이를 설정하고 초과하는 부분을 잘라냅니다.
            corrupt_tokens = corrupt_tokens[:max_length]
            output_tokens = output_tokens[:max_length]
            corrupt_mask_positions = [
                pos if pos < max_length else max_length - 1
                for pos in corrupt_mask_positions
            ]
            urc_inputs = [urc_input[:max_length] for urc_input in urc_inputs]


            """ mlm 입력 """
            batch_corrupt_tokens.append(
                corrupt_tokens
                + [
                    self.tokenizer.pad_token_id
                    for _ in range(max_length - len(corrupt_tokens))
                ]
            )
            batch_mlm_attentions.append(
                [1 for _ in range(len(corrupt_tokens))]
                + [0 for _ in range(max_length - len(corrupt_tokens))]
            )

            """ mlm 출력 """
            batch_output_tokens.append(
                output_tokens
                + [
                    self.tokenizer.pad_token_id
                    for _ in range(max_length - len(corrupt_tokens))
                ]
            )

            """ mlm 레이블 """
            batch_corrupt_mask_positions.append(corrupt_mask_positions)

            """ urc 입력 """
            for urc_input in urc_inputs:
                batch_urc_inputs.append(
                    urc_input
                    + [
                        self.tokenizer.pad_token_id
                        for _ in range(max_length - len(urc_input))
                    ]
                )
                batch_urc_attentions.append(
                    [1 for _ in range(len(urc_input))]
                    + [0 for _ in range(max_length - len(urc_input))]
                )

            """ urc 레이블 """
            batch_urc_labels += urc_labels

        return (
            torch.tensor(batch_corrupt_tokens),
            torch.tensor(batch_output_tokens),
            batch_corrupt_mask_positions,
            torch.tensor(batch_urc_inputs),
            torch.tensor(batch_urc_labels),
            torch.tensor(batch_mlm_attentions),
            torch.tensor(batch_urc_attentions),
        )