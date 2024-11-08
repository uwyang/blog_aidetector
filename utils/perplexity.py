from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import math
import pandas as pd
import numpy as np

# Load the tokenizer and model
model_name="distilgpt2"
perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_perplexity(text):
    """
    Calculate the perplexity of a given text using a specified language model.
    Perplexity is a measure of how well a probability model predicts a sample.
    A lower perplexity indicates the model is better at predicting the sample.
    Args:
        text (str): The input text for which to calculate perplexity.
        model_name (str, optional): The name of the pre-trained language model to use. Defaults to "distilgpt2".
    Returns:
        float: The calculated perplexity of the input text.
    Example:
        text = "The quick brown fox jumps over the lazy dog."
        perplexity = calculate_perplexity(text)
        print(f"Perplexity: {perplexity}")
    """

    inputs = perplexity_tokenizer(text, return_tensors="pt")
    
    # Disable gradient calculation for faster inference
    with torch.no_grad():
        # Get the model's output logits and calculate the log probabilities
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss.item() * inputs['input_ids'].size(1)
        

    # Calculate perplexity
    perplexity = math.exp(log_likelihood / inputs['input_ids'].size(1))
    return perplexity


def calculate_cumulative_perplexity(probabilities, epsilon=1e-9):
    cumulative_perplexities = []
    cumulative_log_sum = 0.0

    for i, prob in enumerate(probabilities):
        cumulative_log_sum += np.log(prob + epsilon)  # Use log to avoid small products
        avg_log_prob = cumulative_log_sum / (i + 1)  # Average log probability up to this point
        perplexity = np.exp(-avg_log_prob)  # Calculate perplexity from the average log
        cumulative_perplexities.append(perplexity)

    return cumulative_perplexities

def results_to_df(results):
    columns = ['i', 'current_token', 'actual_next_token', 
           'logit_mean', 'logit_std', 
           'prob_mean', 'prob_std', 'log_prob_mean', 'log_prob_std', 
           'actual_next_token_logit', 'actual_next_token_prob', 'actual_next_token_log_prob_z_score', 'actual_next_token_rank']

    results_df = pd.DataFrame(results, columns=columns)
    return results_df


def get_sequential_predictions_stats(text, max_tokens = 512):
    """
    Generate sequential prediction statistics for a given text using a pre-trained language model.
    Args:
        text (str): The input text for which to generate prediction statistics.
        max_tokens (int, optional): The maximum number of tokens to consider from the input text. Defaults to 512.
    Returns:
        list: A list of lists where each inner list contains the following statistics for each token in the input text:
            - int: The index of the current token.
            - str: The current token.
            - str: The actual next token.
            - float: The mean of the logits for the next token prediction.
            - float: The standard deviation of the logits for the next token prediction.
            - float: The mean of the probabilities for the next token prediction.
            - float: The standard deviation of the probabilities for the next token prediction.
            - float: The mean of the log probabilities for the next token prediction.
            - float: The standard deviation of the log probabilities for the next token prediction.
            - float: The logit value for the actual next token.
            - float: The probability value for the actual next token.
    """
    # Load model and tokenizer
    tokenizer = perplexity_tokenizer
    model = perplexity_model
    # Initial tokenization of the input text
    tokens = tokenizer.tokenize(text)
    
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    
    #print(f"Initial text: '{text}'\n")
    
    results = []
    # Step through each token in the initial text (except the last token since it has no "next" token)
    

    # Reconstruct the sequence up to the current token
    all_tokens = tokenizer.convert_tokens_to_string(tokens)
    input_ids = tokenizer(all_tokens, return_tensors="pt")["input_ids"]

    # Get model outputs for the current input
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)


    for i in range(len(tokens) - 1):

        # Get logits for the last token (next word prediction)
        last_token_logits = outputs.logits[0, i, :]  # Shape: (vocab_size,)

        # Calculate mean and std of logits
        logit_mean = last_token_logits.mean().item()
        logit_std = last_token_logits.std().item()

        # Calculate probabilities
        probs = F.softmax(last_token_logits, dim=-1)

        # Calculate mean and std of probabilities
        prob_mean = probs.mean().item()
        prob_std = probs.std().item()
        
        # Calculate log probabilities
        log_probs = torch.log(probs + 1e-9)  # Adding a small value to avoid log(0)
        
        # Calculate mean and std of log probabilities
        log_prob_mean = log_probs.mean().item()
        log_prob_std = log_probs.std().item()

        # The actual next token
        actual_next_token = tokens[i+1]
        actual_next_token_id = tokenizer.convert_tokens_to_ids(tokens[i+1])
        actual_next_token_logit = last_token_logits[actual_next_token_id].item()
        actual_next_token_prob = probs[actual_next_token_id].item()
        actual_next_token_log_prob_z_score = (log_probs[actual_next_token_id].item() - log_prob_mean) / log_prob_std
        actual_next_token_rank = (last_token_logits > actual_next_token_logit).sum().item() + 1
        
        # Print statistics and predictions
        current_token = tokens[i]

        results.append([i, current_token, actual_next_token, \
                        logit_mean, logit_std, \
                        prob_mean, prob_std, log_prob_mean, log_prob_std, \
                        actual_next_token_logit, actual_next_token_prob, \
                        actual_next_token_log_prob_z_score, actual_next_token_rank])
    return results
