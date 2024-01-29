import re
import json
import random
from collections import defaultdict


class ConversationProcessor:
    def __init__(self, file_path, pattern=r"\[(.*?)\] \[(.*?)\] (.*)"):
        self.file_path = file_path
        self.pattern = pattern
        self.conversations = self.load_and_process_data()

    def load_and_process_data(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        conversations = []
        for line in lines:
            match = re.match(self.pattern, line)
            if match:
                conversations.append(match.groups())

        return conversations

    def process_chat_combined_speaker(self, conversations):
        processed_data = []
        current_chunk = []
        last_speaker = None
        combined_message = ""

        for i in range(len(conversations)):
            speaker, _, message = conversations[i]
            if speaker != last_speaker:
                if combined_message:
                    current_chunk.append(combined_message)
                combined_message = f"{message}"
                last_speaker = speaker
            else:
                combined_message += f" {message}"
            # When the chunk has 10 messages, add it to the data and start a new chunk
            if len(current_chunk) == 10:
                processed_data.append(current_chunk)
                current_chunk = []

        # Add the current combined message if it's not empty
        if combined_message:
            current_chunk.append(combined_message)

        # Add the last chunk to processed data if it has messages
        if current_chunk:
            processed_data.append(current_chunk)

        return processed_data

    def generate_training_data(self, dataset, output_path, neg_nums=4, use_turns=5):
        train_json = defaultdict(dict)
        all_utts = list({utt for session in dataset for utt in session})
        count = 0

        for session in dataset:
            context = [session[0]]
            for turn in range(1, len(session)):
                utt = session[turn]
                train_json[count]["context"] = context[-use_turns:]
                train_json[count]["positive_response"] = utt
                context.append(utt)

                negative_candidates = random.sample(all_utts, neg_nums)
                train_json[count]["negative_responses"] = negative_candidates
                count += 1

        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(train_json, outfile)


if __name__ == "__main__":
    data_path = "data/processed/all_group.txt"
    processor = ConversationProcessor(data_path)
    session_dataset = processor.process_chat_combined_speaker(processor.conversations)
    train_ratio = 0.8
    split_index = int(len(session_dataset) * train_ratio)
    train_session = session_dataset[:split_index]
    valid_session = session_dataset[split_index:]

    processor.generate_training_data(train_session, "train.json")
    processor.generate_training_data(valid_session, "dev.json")
