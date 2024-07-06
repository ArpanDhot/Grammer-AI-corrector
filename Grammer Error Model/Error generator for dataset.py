import pandas as pd
import json
import random

def generate_random_subset_indices(word):
    length = len(word)
    num_chars = random.randint(1, max(1, length // 2))
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


# Comprehensive error generation with all errors
def generate_all_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map):
    errors = set()
    errors.add(generate_homophone_errors(word, homophones))
    errors.add(generate_phonetic_errors(word, phonetic_map))
    errors.add(generate_keyboard_proximity_errors(word, keyboard_layout))
    errors.add(generate_double_letter_errors(word))
    errors.add(generate_capitalization_errors(word))
    errors.add(generate_word_boundary_errors(word, compound_map))
    errors.add(generate_visual_similarity_errors(word, visual_similarities))
    errors.add(generate_mirror_errors(word))
    errors.add(generate_truncated_errors(word))
    errors.add(generate_repeated_section_errors(word))
    errors.add(generate_common_suffix_prefix_errors(word))
    errors.add(generate_silent_letter_errors(word))
    errors.add(generate_vowel_misplacement_errors(word))
    errors.add(generate_consonant_doubling_dropping_errors(word))
    errors.add(generate_typographic_swap_errors(word, phonetic_map))
    errors.add(generate_plural_possessive_errors(word))
    errors.add(generate_homoglyph_errors(word, visual_similarities))
    errors.add(generate_metathesis_errors(word))
    errors.add(generate_character_omission_errors(word))
    errors.add(generate_character_duplication_errors(word))
    errors.add(generate_whitespace_errors(word))
    errors.add(generate_case_errors(word))
    errors.add(generate_transposition_errors(word))
    errors.add(generate_repetition_errors(word))
    return list(errors)

# List of all error generation functions
error_functions = [
    generate_phonetic_errors,
    generate_homophone_errors,
    generate_keyboard_proximity_errors,
    generate_word_boundary_errors,
    generate_visual_similarity_errors,
    generate_typographic_swap_errors,
    generate_homoglyph_errors,
    generate_common_suffix_prefix_errors,
    generate_silent_letter_errors,
    generate_vowel_misplacement_errors,
    generate_repetition_errors,
    generate_metathesis_errors,
    generate_character_omission_errors,
    generate_character_duplication_errors,
    generate_whitespace_errors,
    generate_case_errors,
    generate_transposition_errors,
    generate_plural_possessive_errors,
    generate_consonant_doubling_dropping_errors,
    generate_mirror_errors,
    generate_truncated_errors,
    generate_repeated_section_errors,
    generate_capitalization_errors,
    generate_double_letter_errors
]


# Function to generate multiple errors
def generate_multiple_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map):
    num_functions = random.randint(2, 8)
    selected_functions = random.sample(error_functions, num_functions)

    errors = set([word])
    for func in selected_functions:
        new_errors = set()
        for error in errors:
            if func == generate_phonetic_errors or func == generate_typographic_swap_errors:
                new_errors.add(func(error, phonetic_map))
            elif func == generate_homophone_errors:
                new_errors.add(func(error, homophones))
            elif func == generate_keyboard_proximity_errors:
                new_errors.add(func(error, keyboard_layout))
            elif func == generate_word_boundary_errors:
                new_errors.add(func(error, compound_map))
            elif func == generate_visual_similarity_errors or func == generate_homoglyph_errors:
                new_errors.add(func(error, visual_similarities))
            else:
                new_errors.add(func(error))
        errors.update(new_errors)

    return list(errors)

# Save results to a CSV file
def save_errors_to_csv(words, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map, output_file):
    data = []
    for word in words:
        all_error_words = generate_all_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map)
        multiple_error_words = generate_multiple_errors(word, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map)
        combined_error_words = set(all_error_words + multiple_error_words)  # Combine both sets of errors
        for error in combined_error_words:
            if error != word:  # Avoid adding the original word as an error
                data.append([word, error])

    df = pd.DataFrame(data, columns=['Words', 'Error Words'])
    df.to_csv(output_file, index=False)


# Usage:

# Load EOWL words
eowl_words_df = pd.read_csv('data/EOWL_words.csv')
eowl_words = eowl_words_df['word'].tolist()

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


save_errors_to_csv(eowl_words, homophones, phonetic_map, keyboard_layout, visual_similarities, compound_map, output_file)
