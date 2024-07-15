
# GrammarAI

GrammarAI is a word error correction system using TensorFlow. This repository contains the code and data required to deploy a pre-trained model for spelling correction. The model uses a tokenizer to preprocess text and provides top predictions with confidence scores for potential corrections.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Detailed Description of Functions](#detailed-description-of-functions)
- [Data](#data)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/GrammarAI.git
    cd GrammarAI
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that the following files are present in the repository:
    - `tokenizer.json`
    - `maxlen.txt`
    - `word_error_correction_model.h5`

2. Run the `model_deployment.py` script to predict spelling corrections:
    ```sh
    python model_deployment.py
    ```

## Files

- `model_deployment.py`: The main script for loading the model, preprocessing input text, making predictions, and displaying results.
- `tokenizer.json`: Contains the tokenizer configuration in JSON format, which maps words to indices.
- `maxlen.txt`: Specifies the maximum sequence length for input text.
- `word_error_correction_model.h5`: The pre-trained TensorFlow model for spelling correction.
- `requirements.txt`: Lists the Python packages required to run the scripts.

## Detailed Description of Functions

### `preprocess_text(text, tokenizer, maxlen)`
This function converts input text into sequences of indices using the tokenizer and pads these sequences to the specified maximum length.

- **Parameters:**
  - `text`: The input text to be processed.
  - `tokenizer`: The tokenizer object used to convert text to sequences.
  - `maxlen`: The maximum length for padding sequences.

- **Returns:** A padded sequence of indices representing the input text.

### `decode_prediction(pred, tokenizer)`
This function decodes the model's prediction (a sequence of indices) back into readable text using the tokenizer's word index.

- **Parameters:**
  - `pred`: The model's prediction output as a sequence of indices.
  - `tokenizer`: The tokenizer object used to convert sequences back to text.

- **Returns:** The decoded text as a string.

### `get_top_n_predictions(pred, tokenizer, n=5)`
This function retrieves the top N predictions for each character in the input text along with their confidence scores.

- **Parameters:**
  - `pred`: The model's prediction output.
  - `tokenizer`: The tokenizer object used to map indices to words.
  - `n`: The number of top predictions to retrieve.

- **Returns:** A list of tuples for each character, each containing the top N predicted words and their confidence scores.

### `predict_spelling_correction(text, model, tokenizer, maxlen, top_n=5)`
This is the main function to process input text, make predictions using the model, and output the top N corrected spelling predictions.

- **Parameters:**
  - `text`: The input text to be corrected.
  - `model`: The pre-trained TensorFlow model.
  - `tokenizer`: The tokenizer object used for text processing.
  - `maxlen`: The maximum length for padding sequences.
  - `top_n`: The number of top predictions to retrieve.

- **Returns:** A list of top N predictions for each character in the input text.

## Data

### Tokenizer
The `tokenizer.json` file contains the configuration for the tokenizer, which includes the word-to-index mapping. This is crucial for converting text into sequences that the model can process.

### Maxlen
The `maxlen.txt` file specifies the maximum length of input sequences. This is used to pad shorter sequences to ensure consistent input size for the model.

### Model
The `word_error_correction_model.h5` file is the pre-trained TensorFlow model that performs the spelling correction.

## Example

To see the model in action, you can use the provided example in `model_deployment.py`:

```python
# Example input text
input_text = "exampel"

# Predict the corrected spelling and get top 5 predictions
top_predictions = predict_spelling_correction(input_text, loaded_model, tokenizer, maxlen, top_n=5)

# Print the top 5 predictions for each character
for i, timestep_predictions in enumerate(top_predictions):
    print(f"Character {i+1}:")
    for word, confidence in timestep_predictions:
        print(f"  {word} (confidence: {confidence:.4f})")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, features, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
