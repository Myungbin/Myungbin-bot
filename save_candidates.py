import pickle
import re

file_path = "data/processed/all_group.txt"

with open(file_path, "r") as file:
    content = file.readlines()

messages_bin = []

for line in content:
    # Checking if the line contains a message from '곽명빈'
    if "[곽명빈]" in line:
        # Extracting the message text
        message = line.split("]")[-1].strip()
        message = re.sub(r"이모티콘|사진|웅|삭제된 메시지입니다.", "", message)
        message = re.sub(r"[^가-힣\s]", "", message)
        messages_bin.append(message)

pickle_file_path = "candidates_new1.pkl"

with open(pickle_file_path, "wb") as file:
    pickle.dump(messages_bin, file)
