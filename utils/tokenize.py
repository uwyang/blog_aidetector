from sentence_transformers import SentenceTransformer, models
import transformers
import re


# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other SentenceTransformer models

# Define tokenizer based on the underlying model of SentenceTransformer
tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

batch_size = 16

def normalize_text(text):
    # Remove any non-printable or non-ASCII characters except common symbols and punctuations
    text = re.sub(r'[^\x20-\x7E]+', '', text)  # Keep characters in the printable ASCII range (space to ~)
    # Normalize spaces (remove extra spaces)
    text = ' '.join(text.split())
    return text

def tokenize_and_truncate(text, normalize=True):
    try:
        if normalize:
            text = normalize_text(text)

        # Tokenize with truncation to handle long sequences
        #print('text')
        #print(text[:100])
        tokens = tokenizer(text, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt', padding=True)
        #print('tokens:')
        #print(tokens[:10])
        truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        return tokens, truncated_text
    except Exception as e:
        print(f"An error occurred during tokenization and truncation: {e}")
        return [], ''

def embed_batch(texts):
    try: 
        tokenization_results = [tokenize_and_truncate(text) for text in texts]
        truncated_texts = [result[1] for result in tokenization_results]
        embeddings = model.encode(truncated_texts, batch_size=batch_size)
        results = []
        for tokenization_result, embedding in zip(tokenization_results, embeddings):
            result = {
                "embedding": embedding,
                "text_embedded": tokenization_result[1], 
                'text_embedding_tc': len(tokenizer.tokenize(tokenization_result[1]))
            }
            results.append(result)
        return results
    except Exception as e:
        print(f"An error occurred during batch embedding: {e}")
        return None
    
def embed_text(text):
    try:
        # Tokenize with truncation to handle long sequences
        tokens, truncated_text = tokenize_and_truncate(text)

        # Generate embeddings for the (truncated) texts
        embeddings = model.encode(truncated_text)

        # Output result as dictionary
        #print(truncated_text)
        result = {
            "embedding": embeddings,
            "text_embedded": truncated_text , 
            'text_embedding_tc': len(tokenizer.tokenize(truncated_text))
        }
        return result
    except Exception as e:
        print(f"An error occurred during text embedding: {e}")
        return None



# Example usage
#texts = ["This is a test.", "Another text to process.", "More texts to embed."]
#process_texts(texts)
