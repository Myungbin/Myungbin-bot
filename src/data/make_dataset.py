import re
from tqdm import tqdm

from src.config import cfg


def process_personal_talk_file(file_path):
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


def save_processed_data(output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for sender, message in process_data:
            file.write(f"[{sender}] {message}\n")


process_data = process_personal_talk_file(cfg.RAW_FILE_PATH)
print(process_data)
