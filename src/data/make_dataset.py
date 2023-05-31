import re
from tqdm import tqdm

from src.config import cfg


def preprocessed_personal_talk_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    messages = re.findall(r'\[([^\]]+)\]\s\[([^\]]+)\]\s(.+)', data)

    processed_data = []
    for sender, _, message in tqdm(messages):
        if sender == cfg.SENDER:
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
    current_message = ""

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
            current_message += " " + message
        else:
            formatted_data.append((current_sender, current_message.strip()))
            current_sender = sender
            current_message = message

    formatted_data.append((current_sender, current_message.strip()))
    return formatted_data


process_data = preprocessed_personal_talk_file(cfg.RAW_FILE_PATH)
save_preprocessed_data(cfg.PROCESSED_FILE_PATH, process_data)
formatted_data = convert_preprocessed_data(cfg.PROCESSED_FILE_PATH)
save_preprocessed_data(cfg.PROCESSED_FILE_PATH, formatted_data)

for sender, message in formatted_data:
    print(f"[{sender}] {message}")
