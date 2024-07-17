import pandas as pd
import transformers
from tqdm import tqdm
import csv
import torch
import ast
import numpy as np
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

transformers.logging.set_verbosity_error()
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# torch.set_num_threads(36)
# Function to split text into chunks of sentences
def split_into_chunks(text, min_len, max_len):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_len:  # +1 for the space or period
            current_chunk += (sentence + '.') if sentence else ''
        else:
            if len(current_chunk) >= min_len:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '.' if sentence else ''
    if len(current_chunk) >= min_len:
        chunks.append(current_chunk.strip())
    return chunks

def split_into_token_chunks(text, tokenizer, max_tokens):
    # Split text into sentences based on periods.
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Add the period back to each sentence except the last one
        if len(sentence) > 0:
            sentence += '.'

        # Tokenize the sentence
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        # Check if adding this sentence would exceed the max length
        if current_length + sentence_length <= max_tokens:
            # Add the sentence tokens to the current chunk
            current_chunk.extend(sentence_tokens)
            current_length += sentence_length
        else:
            # If the current chunk is not empty, save it and start a new chunk
            if current_chunk:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = sentence_tokens
            current_length = sentence_length
       

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks

model_id="NousResearch/Hermes-2-Pro-Llama-3-8B" 
df = pd.read_csv("BTC_match.csv")

output_csv = f"sentiment_{model_id.split('/')[-1]}.csv"

try:
    done_df = pd.read_csv(output_csv)['url']
    df = df[~df['url'].isin(done_df)]
    print("dropped finished")
    
except:
    print("No finished file detected")

df = df.iloc[:]


prompt_template = (
    """<|im_start|>system
    You are an expert in analyzing news about cryptocurrency and Bitcoin. Please provide a sentiment score on how this news will influence the Bitcoin price.
    {attribute}
    On a scale of 1 to 10, 1 stands for very negative, 5 stands for neutral or not related, 10 stands for very positive. Only Provide the number! Remember, only provide a single integer ranging from 1 to 10 without anything else!
    Remember! The First and the only word you should output is a number from 1 to 10!
    <|im_end|>"""

    """<|im_start|>user
    {passage}
    <|im_end|>"""
    """<|im_start|>
    assistant"""
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

#"teknium/OpenHermes-2.5-Mistral-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map='auto' if torch.cuda.is_available() else None
)

model.eval()
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def get_model_responses(messages):
    # Encode the messages and get the attention masks
    encodings = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)#, max_length=model.config.n_positions
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    gen_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "repetition_penalty": 0.1,
    "top_k": 40,
	"num_return_sequences": 1,
	"eos_token_id": tokenizer.eos_token_id,
	"max_new_tokens": 5,
    }
    # Generate responses in batch
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        # max_length=4096,  # or any other generation parameters you want to set
        do_sample=True,
        **gen_config,
    )
    
    # Decode all responses
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses




# New CSV filename



attribute_df = pd.read_csv('prompts.csv')

file_exists = os.path.isfile(output_csv)

with open(output_csv, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["url"] + [aspect for aspect in attribute_df['aspect']])


    with torch.no_grad():
        prompts_count=np.zeros(df.shape[0],dtype = int)
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="generating"):
            responses={}
            article_text = row["article_text"]
            # aspects = ast.literal_eval(row["matched_names"])
            passages = split_into_token_chunks(article_text, tokenizer, 7500) 
            row_prompts_num = 0
            for index,attribute_row in attribute_df.iterrows():
                responses[attribute_row['aspect']] = []
                # for aspect in aspects:
                for passage in passages:
                    prompt = prompt_template.format(attribute=attribute_row['prompt'], passage=passage,)
                    # print(prompt)
                    try:
                        response = get_model_responses([prompt])
                        try:
                            responses[attribute_row['aspect']] = int(response[0].split('assistant')[-1].strip())
                            # print(prompt)
                        except Exception as e:
                            responses[attribute_row['aspect']] = 5
                            # print(response)
                            print(f"Convert failure: {e} original output: {response[0].split('assistant')[-1].strip()}")
                    except Exception as e:
                        print(prompt)
                        print(e)
                        
                        
                    
                    # answers = [n.split('assistant')[-1].strip() for n in response]
                    # print(answers)

            # print(aspects)
            # print(f"prompts_needed:{len(aspects)*len(passages)} actual_prompts:{len(responses)}")
            url=row['url']
            writer.writerow([url] + [responses[aspect] for aspect in attribute_df['aspect']])
            file.flush()

            
        

