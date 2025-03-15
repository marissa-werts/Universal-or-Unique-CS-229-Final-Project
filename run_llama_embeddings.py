import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import argparse

def setup_llama(base_path):
    
    model_path = f"{base_path}/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading model (this might take a few minutes)...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        load_in_8bit=device == "cpu",
        output_hidden_states=True
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def get_embeddings(text, model, tokenizer, layer=-1, pooling='mean'):
    """
    Get embeddings for input text
    """
    if not isinstance(text, str) or not text.strip():
        print(f"Warning: Invalid text input: {text}")
        return np.zeros(model.config.hidden_size)  # Return zero vector for invalid input
    
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # I should change this ! 
    ).to(model.device)
    
    # Get model output with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[layer]
    
    if pooling == 'mean':
        if 'attention_mask' in inputs:
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
            embeddings = torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = embeddings[0]  
        else:
            embeddings = torch.mean(hidden_states[0], dim=0)
    elif pooling == 'first':
        embeddings = hidden_states[0][0]
    else:
        raise ValueError("pooling must be 'mean' or 'first'")
    
    return embeddings.cpu().numpy()

def process_dataframe(df, model, tokenizer, prompt_column='prompts', batch_size=1):
    
    all_embeddings = []
    total_rows = len(df)
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]
        # print(f"Processing prompts {i+1}-{min(i+batch_size, total_rows)} of {total_rows}")
        
        for _, row in batch.iterrows():
            try:
                prompt = str(row[prompt_column])  # Ensure prompt is string
                embedding = get_embeddings(prompt, model, tokenizer)
                all_embeddings.append(embedding)
            except Exception as e:
                # print(f"Error processing prompt: {str(row[prompt_column])[:50]}...")
                # print(f"Error: {str(e)}")
                # # Add zero vector for failed embeddings
                all_embeddings.append(np.zeros(model.config.hidden_size))
    
    # Add embeddings as new column
    df['embeddings'] = [embedding.tolist() for embedding in all_embeddings]
    return df


def main(input_path, output_path, base_path):
    try:
        print("Loading model and tokenizer...")
        model, tokenizer = setup_llama(base_path)
        if '.tsv' in input_path:
            prompts_df = pd.read_csv(input_path, delimiter='\t')
            print("reading in as tsv")
            sep = '\t'
        else:
            prompts_df = pd.read_csv(input_path)
            sep = ','
            print("reading in as csv")
        prompts_df = process_dataframe(prompts_df, model, tokenizer)

        prompts_df.to_csv(output_path, sep=sep, index=False)
        print(f"\nSaved embeddings to {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prompts with Llama model and save embeddings.")
    # parser.add_argument("input_path", help="Path to the input CSV file with prompts.")
    # parser.add_argument("output_path", help="Path to save the output CSV file with embeddings.")
    input_path = '/home/groups/shahrokh/nacohen/Prompts_reasoning.tsv'
    output_path = '/home/groups/shahrokh/nacohen/Prompts_reasoning_emb.tsv'
    # BASE_PATH = "/home/groups/shahrokh/llama_download/llama_model/model_cache/models--meta-llama--Llama-3.1-8B"
    BASE_PATH = "/home/groups/shahrokh/nourya/llama_download/llama_model/model_cache/models--meta-llama--Llama-3.1-8B"
    # parser.add_argument("--base_path", default="/home/groups/shahrokh/llama_download/model_cache/models--meta-llama--Llama-3.1-8B", help="Base path to the Llama model.")

    args = parser.parse_args()
    main(input_path, output_path, base_path = BASE_PATH)