import re
from tqdm import tqdm

from src.config import cfg


# TODO: Q, A의 메세지가 너무 길면 일부 자르는 코드 추가해야됨.

def preprocessed_personal_talk_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    messages = re.findall(r'\[([^\]]+)\]\s\[([^\]]+)\]\s(.+)', data)

    processed_data = []
    for sender, _, message in tqdm(messages):
        if sender in cfg.NB_SENDERS:
            sender = 'Q'
        elif sender == cfg.RECEIVER:
            sender = 'A'
        else:
            continue
        processed_data.append((sender, message))
    return processed_data


def save_preprocessed_data(output_file_path, process_data):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for sender, message in process_data:
            file.write(f"[{sender}] {message}\n")


def convert_preprocessed_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    lines = data.split("\n")
    formatted_data = []
    current_sender = None
    current_message = []
    q_count = 0

    for line in lines:
        if line.startswith("[Q]"):
            sender = "Q"
            message = line[4:].strip()
        elif line.startswith("[A]"):
            sender = "A"
            message = line[4:].strip()
        else:
            continue

        if current_sender is None:
            current_sender = sender

        if current_sender == sender:
            current_message.append(message)
            if sender == "Q":
                q_count += 1
                if q_count > 5:
                    current_message.pop(0)
        else:
            formatted_data.append((current_sender, " ".join(current_message)))
            current_sender = sender
            current_message = [message]
            q_count = 1 if sender == "Q" else 0

    formatted_data.append((current_sender, " ".join(current_message)))
    return formatted_data


process_data = preprocessed_personal_talk_file(cfg.RAW_FILE_PATH)
save_preprocessed_data(cfg.PROCESSED_FILE_PATH, process_data)
formatted_data = convert_preprocessed_data(cfg.PROCESSED_FILE_PATH)
save_preprocessed_data(cfg.PROCESSED_FILE_PATH, formatted_data)
