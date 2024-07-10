import tensorflow as tf
import json
import numpy as np

# Load the tokenizer
with open('tokenizer.json') as f:
    tokenizer_json = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_json))

# Load the maxlen
with open('maxlen.txt') as f:
    maxlen = int(f.read())

# Load the saved model
loaded_model = tf.keras.models.load_model('word_error_correction_model.h5')

# Function to preprocess input text
def preprocess_text(text, tokenizer, maxlen):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding='post')
    return padded_seq

# Function to decode the model's prediction
def decode_prediction(pred, tokenizer):
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    index_to_word[0] = ''  # Add the padding character
    return ''.join([index_to_word.get(idx, '') for idx in pred])

# Function to get top N predictions with their confidence scores
def get_top_n_predictions(pred, tokenizer, n=5):
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    index_to_word[0] = ''  # Add the padding character
    top_n_predictions = []

    for timestep in pred:
        top_indices = np.argsort(timestep)[-n:][::-1]  # Get top N indices
        top_confidences = timestep[top_indices]
        top_words = [index_to_word.get(idx, '') for idx in top_indices]
        top_n_predictions.append(list(zip(top_words, top_confidences)))

    return top_n_predictions

# Predict spelling corrections and get top 5 predictions
def predict_spelling_correction(text, model, tokenizer, maxlen, top_n=5):
    preprocessed_text = preprocess_text(text, tokenizer, maxlen)
    pred = model.predict(preprocessed_text)
    top_predictions = get_top_n_predictions(pred[0], tokenizer, top_n)
    return top_predictions

# Example input text
input_text = "exampel"

# Predict the corrected spelling and get top 5 predictions
top_predictions = predict_spelling_correction(input_text, loaded_model, tokenizer, maxlen, top_n=5)

# Print the top 5 predictions for each character
for i, timestep_predictions in enumerate(top_predictions):
    print(f"Character {i+1}:")
    for word, confidence in timestep_predictions:
        print(f"  {word} (confidence: {confidence:.4f})")
