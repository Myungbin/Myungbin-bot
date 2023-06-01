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