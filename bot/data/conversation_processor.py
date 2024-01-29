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

    def merge_messages(self, min_length=5, max_total_length=100, short_message_merge=False):
        reformatted_data = []
        conversation = []
        current_message = ""

        for i, (_, _, message) in enumerate(self.conversations):
            current_message = f"{current_message} {message}".strip()

            if len(current_message.split()) >= min_length or i == len(self.conversations) - 1:
                if len(current_message) > max_total_length and not short_message_merge:
                    words = current_message.split()
                    trimmed_message = ' '.join(words[:min_length])
                    conversation.append(trimmed_message)
                    current_message = ' '.join(words[min_length:])
                else:
                    conversation.append(current_message)
                    current_message = ""

            if len(conversation) == 10:
                reformatted_data.append(conversation)
                conversation = []

        if conversation:
            reformatted_data.append(conversation)

        return reformatted_data

    def generate_training_data(self, dataset, output_path, neg_nums=4, use_turns=5):
        train_json = defaultdict(dict)
        all_utts = list({utt for session in dataset for utt in session})
        count = 0

        for session in dataset:
            context = [session[0]]
            for turn in range(1, len(session)):
                utt = session[turn]
                train_json[count]['context'] = context[-use_turns:]
                train_json[count]['positive_response'] = utt
                context.append(utt)

                negative_candidates = random.sample(all_utts, neg_nums)
                train_json[count]['negative_responses'] = negative_candidates
                count += 1

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(train_json, outfile)

if __name__ == "__main__":
    data_path = "data/raw/all_group.txt"
    processor = ConversationProcessor(data_path)
    session_dataset = processor.merge_messages()

    train_ratio = 0.8
    split_index = int(len(session_dataset) * train_ratio)
    train_session = session_dataset[:split_index]
    valid_session = session_dataset[split_index:]

    processor.generate_training_data(train_session, 'train.json')
    processor.generate_training_data(valid_session, 'dev.json')
