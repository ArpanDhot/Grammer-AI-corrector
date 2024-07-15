
# Grammer-AI-corrector

GrammarAI is a deep learning-based project aimed at correcting spelling and grammatical errors in text. It leverages TensorFlow for model training and prediction, using a sequence-to-sequence approach with LSTM layers.



## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Training the Model](#training-the-model)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [Files](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/grammarAI.git
    cd grammarAI
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data

The dataset used for training the model is located in the `data` directory. The dataset consists of pairs of incorrect and correct sentences, which are used to train the sequence-to-sequence model for error correction.

- `data/sentences.txt`: This file contains sentences with spelling and grammatical errors.
- `data/corrected_sentences.txt`: This file contains the correct versions of the sentences in `sentences.txt`.

## Training the Model

The training script is `train.py`, which prepares the data, builds the model, and trains it.

### Steps in `train.py`:

1. **Prepare the data:**
    - **Tokenization**: Convert sentences into sequences of integers.
    - **Padding**: Pad sequences to a maximum length to ensure uniform input size for the model.

    ```python
    # Tokenizing and padding the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')
    ```

2. **Build the model:**
    - **Embedding Layer**: Converts input words into dense vectors of fixed size.
    - **LSTM Layers**: Capture temporal dependencies in the sequence data.
    - **Dense Layer**: Applies a softmax activation to predict the next word in the sequence.

    ```python
    # Building the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ```

3. **Train the model:**
    ```bash
    python train.py
    ```
    This script saves:
    - **`word_error_correction_model.h5`**: The trained model.
    - **`tokenizer.json`**: The tokenizer used for converting text to sequences.
    - **`maxlen.txt`**: The maximum length of sequences, used for padding during preprocessing.

## Model Deployment

The `deployment.py` script loads the trained model and tokenizer, preprocesses input text, and predicts the corrected text.

### Functions in `deployment.py`:

- **`preprocess_text`**: Converts input text to sequences and pads them to the maximum length.

    ```python
    def preprocess_text(text, tokenizer, maxlen):
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding='post')
        return padded_seq
    ```

- **`decode_prediction`**: Converts the model's numerical predictions back to words using the tokenizer's word index.

    ```python
    def decode_prediction(pred, tokenizer):
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}
        index_to_word[0] = ''  # Add the padding character
        return ''.join([index_to_word.get(idx, '') for idx in pred])
    ```

- **`get_top_n_predictions`**: Retrieves the top N predictions for each word with their confidence scores.

    ```python
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
    ```

- **`predict_spelling_correction`**: Main function to predict corrections for input text.

    ```python
    def predict_spelling_correction(text, model, tokenizer, maxlen, top_n=5):
        preprocessed_text = preprocess_text(text, tokenizer, maxlen)
        pred = model.predict(preprocessed_text)
        top_predictions = get_top_n_predictions(pred[0], tokenizer, top_n)
        return top_predictions
    ```

## Usage

To use the model for spelling correction, run the `deployment.py` script. This script takes an input text, preprocesses it, predicts the corrected text, and prints the top 5 predictions for each character.

Example:
```bash
python deployment.py
```

Output:
```
Character 1:
  e (confidence: 0.9837)
  a (confidence: 0.0123)
  i (confidence: 0.0021)
  ...
```


## Directory Structure

```plaintext
GrammarAI/
│
├── data/
│   ├── EOWL_words.csv
│   ├── compound_map.json
│   ├── homophones_map.json
│   ├── keyboard_layout_map.json
│   ├── phonetic_map.json
│   └── visual_similarities_map.json
│
├── preprocessing.py
├── training.py
├── deployment.py
├── tokenizer.json
├── maxlen.txt
├── word_error_correction_model.h5
└── README.md
```

## Dataset Files

The data directory contains several important files used for training the model and providing additional mappings and configurations:

- **[EOWL_words.csv](path/to/EOWL_words.csv)**: A CSV file containing a list of English words used for training and evaluating the model.
- **[compound_map.json](path/to/compound_map.json)**: A JSON file containing mappings for compound words.
- **[homophones_map.json](path/to/homophones_map.json)**: A JSON file mapping words to their homophones.
- **[keyboard_layout_map.json](path/to/keyboard_layout_map.json)**: A JSON file representing the keyboard layout to assist in understanding common typing errors.
- **[phonetic_map.json](path/to/phonetic_map.json)**: A JSON file containing phonetic mappings of words to assist in correcting phonetically similar errors.
- **[visual_similarities_map.json](path/to/visual_similarities_map.json)**: A JSON file mapping visually similar characters to help in correcting visually similar mistakes.

### Data Files

- **`data/sentences.txt`**: Contains sentences with spelling and grammatical errors.
- **`data/corrected_sentences.txt`**: Contains the correct versions of the sentences in `sentences.txt`.

### Script Files

- **`train.py`**: Script to train the model.
  - **Tokenization and Padding**: Converts and pads sentences to uniform length.
  - **Model Building**: Defines the LSTM-based sequence-to-sequence model.
  - **Model Training**: Trains the model and saves the trained model, tokenizer, and maximum sequence length.

    ```python
    # Training the model
    model.fit(padded_sequences, padded_corrected_sequences, epochs=10, batch_size=64, validation_split=0.2)
    model.save('word_error_correction_model.h5')
    
    # Save tokenizer and maxlen
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer.to_json())
    with open('maxlen.txt', 'w') as f:
        f.write(str(maxlen))
    ```

- **`deployment.py`**: Script to deploy the model and make predictions.
  - **Loading Resources**: Loads the trained model, tokenizer, and maximum sequence length.
  - **Preprocessing**: Converts input text to padded sequences.
  - **Prediction and Decoding**: Predicts corrections and decodes them into readable text.

### Model and Tokenizer Files

- **`word_error_correction_model.h5`**: The trained model file saved after running `train.py`.
- **`tokenizer.json`**: The tokenizer configuration saved after training.
- **`maxlen.txt`**: The maximum sequence length used during training.



## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
