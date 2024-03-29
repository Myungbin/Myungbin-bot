# Myungbin-bot


  ![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)   

`Myungbin-bot v1.0.0`  


### Overview

Myungbin-bot is a model designed for processing conversational data and generating dialogue responses using deep learning techniques. The focus of this project is to analyze chat sessions and provide meaningful responses. Based on PyTorch and the Transformers library by Hugging Face, it utilizes the Klue/Roberta-based model.

### Data Processing

In the Myungbin-bot project, we utilize a specific format for conversational data, akin to the structure found in KakaoTalk chat logs. The data preprocessing steps are vital for ensuring the model works effectively with real-world conversational patterns. Below are key aspects of our data processing approach:

#### Data Loading and Extraction
The method `load_and_process_data` is crucial for reading and extracting conversation data from the input files. It operates as follows:

1. **Reading the File**: The method reads a file line by line, decoding each line in UTF-8 format.
2. **Regex Pattern Matching**: Each line is parsed using a regular expression pattern `r"\[(.*?)\] \[(.*?)\] (.*)"`. This pattern extracts the speaker, timestamp, and message from each line, mimicking the typical KakaoTalk message format.
3. **Data Structuring**: The extracted data are structured into tuples of (speaker, time, message) and stored in a list named `conversations`.

#### PreprocessChat
The `preprocess_chat` static method performs three critical functions:

1. **Speaker Segmentation**: Differentiates messages based on the speaker, ensuring that each conversation chunk is attributed to the correct individual.
2. **Conversation Chunking**: Organizes messages into chunks for easier processing, with a default chunk size of 10 messages.
3. **Message Combination**: Concatenates consecutive messages from the same speaker into a single message for contextual coherence.

These preprocessing steps are fundamental in transforming raw chat logs into a structured format suitable for the deep learning models employed in the Myungbin-bot project. By replicating the nuances of real-world conversations, the model can generate more accurate and contextually relevant responses.


## Model Description



### PostModel

#### Architecture
- **RobertaForMaskedLM**: Utilizes a pretrained `klue/roberta-base` model, specifically designed for masked language modeling (MLM) tasks. This component is crucial for understanding and predicting masked tokens within a sentence, which is a fundamental part of language understanding.
- **Hidden Dimension**: Extracts the hidden size from the Roberta model's configuration, used for downstream tasks.
- **Tokenizer**: Uses the `AutoTokenizer` from the same `klue/roberta-base` model. It includes special token handling (`<SEP>`) and is configured for a maximum length of 512 tokens.



### FineModel

#### Architecture
- **RobertaForMaskedLM**: Similar to `PostModel`, it uses the `klue/roberta-base` model. This consistency ensures that both models operate under a similar understanding of language and context.
- **Hidden Dimension**: Same as in `PostModel`, it defines the size of the hidden layer.
- **Tokenizer**: Configured identically to `PostModel`, ensuring consistency in tokenization across models.


These models play a pivotal role in enabling Myungbin-bot to process, understand, and generate responses based on conversational data. `PostModel` is primarily used for understanding and predicting elements within a conversation, while `FineModel` is tailored towards classifying or making binary decisions about the processed data. The integration of these models allows Myungbin-bot to handle complex conversational scenarios effectively.

### Inference Process
After setting up and training Myungbin-bot, you can use it to evaluate and select the most appropriate responses for a given context. The following step illustrates how to load a trained model, process conversation data, and determine the best response candidates. This step assumes you have a pre-trained model and a set of candidate responses.

#### Running the Evaluation and Selection Process

1. **Loading Candidate Responses**: The `load_candidates` function reads a list of potential responses from a pickle file, ensuring no duplicates.

2. **Tokenizing the Context**: `tokenize_context` tokenizes the conversation context using the model's tokenizer.

3. **Evaluating Candidates**: The `evaluate_candidates` function takes the tokenized context and a list of response candidates, and evaluates each candidate's suitability using the model. It processes the candidates in batches for efficiency.

4. **Selecting Top Candidates**: Finally, `select_top_candidates` ranks the evaluated candidates based on their suitability scores and selects the top ones.
