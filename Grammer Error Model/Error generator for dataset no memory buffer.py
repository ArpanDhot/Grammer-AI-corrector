import pandas as pd
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# Global queue and lock for thread-safe operations
task_queue = Queue()
status_lock = threading.Lock()

def generate_random_subset_indices(word):
    length = len(word)
    if length == 0:
        return []  # Handle empty word case
    num_chars = random.randint(1, max(1, length // 2))
    if num_chars > length:
        num_chars = length
    return random.sample(range(length), num_chars)


# Define error generation functions
def generate_phonetic_errors(word, phonetic_map):
    errors = set()
    for i, char in enumerate(word):
        if char in phonetic_map:
            for substitute in phonetic_map[char]:
                errors.add(word[:i] + substitute + word[i + 1:])
    return list(errors)

def generate_homophone_errors(word, homophones):
    return homophones.get(word, [])

def generate_keyboard_proximity_errors(word, keyboard_layout):
    errors = set()
    for i, char in enumerate(word):
        if char in keyboard_layout:
            for adjacent in keyboard_layout[char]:
                errors.add(word[:i] + adjacent + word[i + 1:])
    return list(errors)

def generate_word_boundary_errors(word, compound_map):
    errors = set()
    for comp in compound_map:
        if word in comp.split():
            errors.add(comp)
    return list(errors)

def generate_visual_similarity_errors(word, visual_similarities):
    errors = set()
    for i, char in enumerate(word):
        if char in visual_similarities:
            for substitute in visual_similarities[char]:
                errors.add(word[:i] + substitute + word[i + 1:])
    return list(errors)

def generate_typographic_swap_errors(word, phonetic_map):
    mistakes = set()
    for sequence in phonetic_map:
        if sequence in word:
            for swap in phonetic_map[sequence]:
                mistakes.add(word.replace(sequence, swap))
    return list(mistakes)

def generate_homoglyph_errors(word, visual_similarities):
    mistakes = set()
    for i, char in enumerate(word):
        if char in visual_similarities:
            for homoglyph in visual_similarities[char]:
                mistakes.add(word[:i] + homoglyph + word[i+1:])
    return list(mistakes)

def generate_common_suffix_prefix_errors(word):
    prefixes = [
        'un', 're', 'in', 'im', 'il', 'ir', 'dis', 'en', 'em', 'non', 'over', 'mis', 'sub', 'pre',
        'inter', 'fore', 'de', 'trans', 'super', 'semi', 'anti', 'mid', 'under'
    ]
    suffixes = [
        'able', 'ible', 'al', 'ial', 'ed', 'en', 'er', 'or', 'est', 'ful', 'ic', 'ing', 'ion', 'tion',
        'ation', 'ition', 'ity', 'ty', 'ive', 'ative', 'itive', 'less', 'ly', 'ment', 'ness', 'ous',
        'eous', 'ious', 's', 'es', 'y'
    ]
    mistakes = set()

    # Check for prefixes
    for prefix in prefixes:
        if word.startswith(prefix):
            without_prefix = word[len(prefix):]
            # Add or remove the prefix
            mistakes.add(without_prefix)
            mistakes.add(prefix + prefix + without_prefix)

    # Check for suffixes
    for suffix in suffixes:
        if word.endswith(suffix):
            without_suffix = word[:-len(suffix)]
            # Add or remove the suffix
            mistakes.add(without_suffix)
            mistakes.add(without_suffix + suffix + suffix)

    return list(mistakes)

def generate_silent_letter_errors(word):
    silent_letter_rules = ['e', 'k', 'w', 'h']
    mistakes = set()
    for letter in silent_letter_rules:
        if letter in word:
            mistakes.add(word.replace(letter, ''))
    return list(mistakes)

def generate_vowel_misplacement_errors(word):
    vowels = 'aeiou'
    mistakes = set()
    for i, char in enumerate(word):
        if char in vowels:
            for vowel in vowels:
                if vowel != char:
                    mistakes.add(word[:i] + vowel + word[i+1:])
    return list(mistakes)


# Error generation variation two
def generate_repetition_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    # Repeat part of the word
    for i in indices:
        mistakes.add(word[:i] + word[:i] + word[i:])
    # Repeat the whole word
    mistakes.add(word + word)
    return list(mistakes)

def generate_metathesis_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    for i in indices:
        for j in range(i + 2, len(word)):
            transposed = list(word)
            transposed[i], transposed[j] = transposed[j], transposed[i]
            mistakes.add(''.join(transposed))
    return list(mistakes)

def generate_character_omission_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    for i in indices:
        mistakes.add(word[:i] + word[i + 1:])
    return list(mistakes)

def generate_character_duplication_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    for i in indices:
        mistakes.add(word[:i] + word[i] + word[i] + word[i + 1:])
    return list(mistakes)

def generate_whitespace_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    # Adding spaces
    for i in indices:
        if i < len(word):
            mistakes.add(word[:i] + ' ' + word[i:])
    # Removing spaces
    mistakes.add(word.replace(' ', ''))
    return list(mistakes)

def generate_case_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    for i in indices:
        if word[i].isalpha():
            swapped_case = word[:i] + word[i].swapcase() + word[i + 1:]
            mistakes.add(swapped_case)
    return list(mistakes)

def generate_transposition_errors(word):
    indices = generate_random_subset_indices(word)
    mistakes = set()
    for i in indices:
        if i < len(word) - 1:
            swapped = list(word)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            mistakes.add(''.join(swapped))
    return list(mistakes)

def generate_plural_possessive_errors(word):
    if len(word) > 1:
        mistakes = set()
        if word.endswith('s'):
            mistakes.add(word[:-1])
        else:
            mistakes.add(word + 's')
        return list(mistakes)
    return []

def generate_consonant_doubling_dropping_errors(word):
    indices = generate_random_subset_indices(word)
    consonants = 'bcdfghjklmnpqrstvwxyz'
    mistakes = set()
    for i in indices:
        char = word[i]
        if char in consonants:
            # Double the consonant
            mistakes.add(word[:i] + char + char + word[i + 1:])
            # Drop the consonant
            if i < len(word) - 1 and word[i + 1] == char:
                mistakes.add(word[:i] + word[i + 1:])
    return list(mistakes)

def generate_mirror_errors(word):
    if len(word) > 1:
        return [word[::-1]]
    return []

def generate_truncated_errors(word):
    if len(word) > 1:
        return [word[:i] for i in range(1, len(word))]
    return []

def generate_repeated_section_errors(word):
    if len(word) > 1:
        return [word + word]
    return []

def generate_capitalization_errors(word):
    indices = generate_random_subset_indices(word)
    errors = set()
    for i in indices:
        if word[i].isalpha():
            errors.add(word[:i] + word[i].swapcase() + word[i + 1:])
    return list(errors)

def generate_double_letter_errors(word):
    indices = generate_random_subset_indices(word)
    errors = set()
    for i in indices:
        if i < len(word) - 1 and word[i] == word[i + 1]:
            errors.add(word[:i + 1] + word[i + 2:])  # Omit double letter
    for i in indices:
        errors.add(word[:i] + word[i] + word[i:])  # Add double letter
    return list(errors)


#----------------------------------------------------------------------------------------------------------------------

# Comprehensive error generation with all errors
def generate_all_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map):
    errors = set()
    errors.update(generate_homophone_errors(word, homophones))
    errors.update(generate_phonetic_errors(word, phonetic_map))
    errors.update(generate_keyboard_proximity_errors(word, keyboard_layout))
    errors.update(generate_double_letter_errors(word))
    errors.update(generate_capitalization_errors(word))
    errors.update(generate_word_boundary_errors(word, compound_map))
    errors.update(generate_visual_similarity_errors(word, visual_similarities))
    errors.update(generate_mirror_errors(word))
    errors.update(generate_truncated_errors(word))
    errors.update(generate_repeated_section_errors(word))
    errors.update(generate_common_suffix_prefix_errors(word))
    errors.update(generate_silent_letter_errors(word))
    errors.update(generate_vowel_misplacement_errors(word))
    errors.update(generate_consonant_doubling_dropping_errors(word))
    errors.update(generate_typographic_swap_errors(word, phonetic_map))
    errors.update(generate_plural_possessive_errors(word))
    errors.update(generate_homoglyph_errors(word, visual_similarities))
    errors.update(generate_metathesis_errors(word))
    errors.update(generate_character_omission_errors(word))
    errors.update(generate_character_duplication_errors(word))
    errors.update(generate_whitespace_errors(word))
    errors.update(generate_case_errors(word))
    errors.update(generate_transposition_errors(word))
    errors.update(generate_repetition_errors(word))
    return list(errors)


# Function to process a single word and save results to the CSV
def process_word(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map, output_file):
    all_error_words = generate_all_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities,
                                          compound_map)

    combined_error_words = set(all_error_words)  # Combine both sets of errors
    data = [[word, error] for error in combined_error_words if
            error != word]  # Avoid adding the original word as an error

    df = pd.DataFrame(data, columns=['Words', 'Error Words'])
    df.to_csv(output_file, mode='a', index=False, header=False)

    with status_lock:
        task_queue.get()
        print_status(task_queue.qsize(), total_tasks, max_workers)


# Function to handle multi-threaded processing and saving
def save_errors_to_csv(words, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map, output_file, max_workers=100):
    global total_tasks
    total_tasks = len(words)

    # Initialize the output file with headers
    df = pd.DataFrame(columns=['Words', 'Error Words'])
    df.to_csv(output_file, mode='w', index=False, header=True)

    # Add tasks to the queue
    for word in words:
        task_queue.put(word)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_word, word, homophones, phonetic_map, keyboard_layout, visual_similarities,
                                   compound_map, output_file) for word in words]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Generated an exception: {exc}')


def print_status(remaining_tasks, total_tasks, max_workers):
    completed_tasks = total_tasks - remaining_tasks
    busy_threads = max_workers - remaining_tasks if remaining_tasks < max_workers else max_workers
    percentage_done = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    print(
        f"Total tasks: {total_tasks}, Completed: {completed_tasks}, Busy threads: {busy_threads}, Percentage done: {percentage_done:.2f}%")


# Usage:

# Load EOWL words
eowl_words_df = pd.read_csv('data/EOWL_words.csv')
eowl_words = eowl_words_df['Words'].tolist()

# Load homophones map
with open('data/homophones_map.json') as f:
    homophones = json.load(f)

# Load keyboard layout map
with open('data/keyboard_layout_map.json') as f:
    keyboard_layout = json.load(f)

# Load phonetic map
with open('data/phonetic_map.json') as f:
    phonetic_map = json.load(f)

# Load visual similarities map
with open('data/visual_similarities_map.json') as f:
    visual_similarities = json.load(f)

# Load compound words map
with open('data/compound_map.json') as f:
    compound_map = json.load(f)

output_file = 'errors.csv'
max_workers = 100

save_errors_to_csv(eowl_words, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map,
                   output_file, max_workers=max_workers)