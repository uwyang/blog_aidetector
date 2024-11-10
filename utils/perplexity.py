from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
import math
import pandas as pd
import numpy as np

# Load the tokenizer and model
model_name="distilgpt2"
#perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
#perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)



def calculate_perplexity(text, model, tokenizer):
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
    #reset model to re-initialize the state. 
    #perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
    #perplexity_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    perplexity_tokenizer = model
    perplexity_model = tokenizer


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


def calculate_windowed_perplexity(probabilities, epsilon=1e-9, window=10):

    problog = [np.log(prob + epsilon) for prob in probabilities]
    meanproblog = [np.mean(problog[max(0, i - window):i]) for i in range(1, len(problog) + 1)]
    perplexities = [np.exp(-logprob) for logprob in meanproblog]

    return perplexities



def results_to_df(results):
    columns = ['i', 'current_token', 'actual_next_token', 
           'logit_mean', 'logit_std', 
           'prob_mean', 'prob_std', 'log_prob_mean', 'log_prob_std', 
           'actual_next_token_logit', 'actual_next_token_prob', 'actual_next_token_log_prob_z_score', 
           'actual_next_token_prob_z_score', 'actual_next_token_logit_z_score', 'actual_next_token_rank']

    results_df = pd.DataFrame(results, columns=columns)
    return results_df

def evaluate(model, input_ids):
    # Get model outputs for the current input
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    
    #loss of the model: 
    loss = outputs.loss.item()

    # Logits from the model
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    return loss, logits


def get_eval_results(text,  model, tokenizer,max_tokens = 512):
    """
    Generate sequential prediction statistics for a given text using a pre-trained language model.
    Args:
        text (str): The input text for which to generate prediction statistics.
        max_tokens (int, optional): The maximum number of tokens to consider from the input text. Defaults to 512.
    Returns:

    """
    # Load model and tokenizer, reset all parameters
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


    # Initial tokenization of the input text
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
        input_ids = input_ids[:, :max_tokens]
    
    

    # Get model outputs for the current input
    loss, logits = evaluate(model, input_ids)
    return logits, tokens, input_ids, loss

def get_sequential_predictions_statsdf(logits, tokens, input_ids, tokenizer):
    # Reshape logits for cross-entropy calculation
    batch_size, sequence_length, vocab_size = logits.size()
    # Calculate per-token cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fn(logits.view(-1, vocab_size), input_ids.view(-1))

    per_token_loss = per_token_loss.view(sequence_length)
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)





    # Shift input_ids to get the actual next token at each position
    next_tokens = input_ids[:, 1:]  # Drop the first token for next-token prediction
    pred_probs = probs[:, :-1, :]  # Align with next tokens, ignore last position

    # Gather the probabilities of the actual next tokens
    next_token_probs = torch.gather(pred_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    # Calculate the rank of the actual next token in the predicted probabilities
    next_token_ranks = torch.argsort(pred_probs, dim=-1, descending=True)
    next_token_rank = (next_token_ranks == next_tokens.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
    # Calculate mean logits for each token
    actual_next_token_logits = torch.gather(logits[:, :-1, :], -1, next_tokens.unsqueeze(-1)).squeeze(-1)


    # Calculate z-scores for logits
    logit_mean = logits.mean(dim=-1, keepdim=True)
    logit_std = logits.std(dim=-1, keepdim=True)
    #logit_z_scores = (logits - logit_mean) / (logit_std + 1e-9)  # Add epsilon to avoid division by zero
    # Drop the last item in logit_mean to line up with actual_next_token_logits
    actual_next_token_logit_z_scores = (actual_next_token_logits.squeeze() - logit_mean.squeeze()[:-1]) / (logit_std.squeeze()[:-1] + 1e-9)

    # Calculate z-scores for probabilities
    prob_mean = probs.mean(dim=-1, keepdim=True)
    prob_std = probs.std(dim=-1, keepdim=True)
    actual_next_token_probs = torch.gather(probs[:, :-1, :], -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    #prob_z_scores = (probs - prob_mean) / (prob_std + 1e-9)  # Add epsilon to avoid division by zero
    # Calculate z-scores for actual next token probabilities
    actual_next_token_prob_z_scores = (actual_next_token_probs - prob_mean.squeeze()[:-1]) / (prob_std.squeeze()[:-1] + 1e-9)

    # Calculate the most probable next token and its probability
    most_probable_next_token_id = torch.argmax(pred_probs, dim=-1).squeeze().tolist()
    most_probable_next_token_prob = torch.max(pred_probs, dim=-1).values.squeeze().tolist()
    most_probable_next_token = tokenizer.convert_ids_to_tokens(most_probable_next_token_id)


    # Define a function to convert a tensor or list to a padded 1D numpy array
    # padding with last element. 
    def to_padded_numpy_array(var, target_length):
        if isinstance(var, torch.Tensor):
            var = var.cpu().numpy().flatten()  # Convert to 1D numpy array
        elif isinstance(var, list):
            var = np.array(var).flatten()  # Convert to 1D numpy array if it's a list
        
        if len(var) < target_length:
            # Pad with the last element to reach target_length
            var = np.pad(var, (0, target_length - len(var)), 'edge')
        return var

    # Assuming each variable is defined as shown
    # Variables that need to be padded
    next_tokens = to_padded_numpy_array(next_tokens, len(tokens))
    next_token_probs = to_padded_numpy_array(next_token_probs, len(tokens))
    next_token_rank = to_padded_numpy_array(next_token_rank, len(tokens))
    actual_next_token_logits = to_padded_numpy_array(actual_next_token_logits, len(tokens))
    actual_next_token_logit_z_scores = to_padded_numpy_array(actual_next_token_logit_z_scores, len(tokens))
    actual_next_token_probs = to_padded_numpy_array(actual_next_token_probs, len(tokens))
    actual_next_token_prob_z_scores = to_padded_numpy_array(actual_next_token_prob_z_scores, len(tokens))
    most_probable_next_token_id = to_padded_numpy_array(most_probable_next_token_id, len(tokens))
    most_probable_next_token_prob = to_padded_numpy_array(most_probable_next_token_prob, len(tokens))
    most_probable_next_token = to_padded_numpy_array(most_probable_next_token, len(tokens))

    # Variables with length 23
    input_ids = to_padded_numpy_array(input_ids, len(tokens))
    logit_mean = to_padded_numpy_array(logit_mean, len(tokens))
    logit_std = to_padded_numpy_array(logit_std, len(tokens))
    prob_mean = to_padded_numpy_array(prob_mean, len(tokens))
    prob_std = to_padded_numpy_array(prob_std, len(tokens))

    variables = [
        tokens, 
        input_ids, 
        tokens[1:]+[None],
        next_tokens,
        next_token_probs,
        next_token_rank,
        actual_next_token_logits,
        actual_next_token_logit_z_scores,
        actual_next_token_probs,
        actual_next_token_prob_z_scores,
        most_probable_next_token_id,
        most_probable_next_token_prob,
        most_probable_next_token,
        logit_mean,
        logit_std,
        prob_mean,
        prob_std
    ]
    df = pd.DataFrame(variables).T

    df.columns = [
        "tokens",
        "input_id",
        "next_token",
        "next_input_id",
        "next_token_probs",
        "next_token_rank",
        "actual_next_token_logits",
        "actual_next_token_logit_z_scores",
        "actual_next_token_probs",
        "actual_next_token_prob_z_scores",
        "most_probable_next_token_id",
        "most_probable_next_token_prob",
        "most_probable_next_token",
        "logit_mean",
        "logit_std",
        "prob_mean",
        "prob_std"]

    df['perplexity'] = calculate_cumulative_perplexity(df['actual_next_token_probs'])

    return df
