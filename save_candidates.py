import pickle

file_path = 'data/raw/all_group.txt'

with open(file_path, 'r') as file:
    content = file.readlines()
    
messages_bin = []

for line in content:
    # Checking if the line contains a message from '곽명빈'
    if '[곽명빈]' in line:
        # Extracting the message text
        message = line.split(']')[-1].strip()
        messages_bin.append(message)

pickle_file_path = 'candidates.pkl'

# with open(pickle_file_path, 'wb') as file:
#     pickle.dump(messages_bin, file)