# AI Language Detection with Language Checker

This project provides a Python module for detecting the language of a given text using AI via the FastText language identification model by Meta-Facebook. It includes utilities for mapping between ISO 639-3 language codes and their corresponding language names, as well as a wrapper for easier use of the AI model.

## Overview

The module consists of two main classes:

- **LanguageCodes**: Handles mapping between ISO 639-3 language codes and their corresponding language names.
- **LanguageChecker**: Uses the FastText language identification model to detect the language of a given text.


## Features

- **Language Detection**: Detects the language of input text with high accuracy.
- **Confidence Scores**: Provides confidence scores for predictions.
- **Language Mapping**: Converts between ISO 639-3 language codes and language names.
- **Top K Predictions**: Retrieves the top K language candidates for a given text.
- **Exception Handling**: Raises exceptions for low-confidence predictions when a certainty threshold is specified.

## Installation

If you have git installed, you can easily install via pip:

```bash
pip install git+https://github.com/laelhalawani/language_checker.git
```

Otherwise download the repo, cd into the top level language checker folder and install via pip:

```bash
pip install .
```

From PyPI not yet available, will become available soon with first release.

## Usage

The `LanguageChecker` class provides various methods for language detection and comparison. Here are some examples of how to use it:

### Basic Usage

```python
from language_checker import LanguageChecker

checker = LanguageChecker()

# Simple language prediction
text_en = "Hello, how are you?"
language_name = checker.predict_language(text_en)
print(f"Predicted language: {language_name}")
# Output: Predicted language: English
```

### Language Prediction with Certainty Threshold

```python
# Language prediction with defined minimum acceptable certainty
text_mixed = "Gut morning, jak are du?"
try:
    language_name = checker.predict_language(text_mixed, certainty=0.999)
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: Language detection confidence 0.86 is below the threshold of 1.00.
```

### Language Prediction with Confidence Score

```python
# Language prediction with outputting the model's prediction confidence
language_name, confidence = checker.predict_language_and_certainty(text_en)
print(f"Predicted language: {language_name} with confidence: {confidence:.2f}")
# Output: Predicted language: English with confidence: 1.00
```

### Getting Language Candidates

```python
# Getting language candidates with confidence scores
candidates = checker.predict_language_candidates(text_en, k=3)
print("Language candidates:")
for name, confidence in candidates:
    print(f"\t{name}: {confidence:.6f}")
# Output:
# Language candidates:
#     English: 1.000006
#     Italian: 0.000011
#     Romanian: 0.000011
```

### Checking Specific Language

```python
# Checking if a text is in a specific language
is_en = checker.is_language("english", text_en)
print(f"Is text confirmed to be in English: {is_en}")
# Output: Is text confirmed to be in English: True

# Checking with a high certainty threshold for a mixed language text
is_en = checker.is_language("english", text_mixed, certainty=0.999)
print(f"Is text confirmed to be in English: {is_en}")
# Output:
# WARNING:root:Failed to predict language for text: Gut morning, jak are du?, returning False.
# Is text confirmed to be in English: False
```

### Comparing Multiple Texts

```python
# Checking if multiple texts are in the same language
text_en_2 = "Hi, I'm fine. How are you?"
text_pl = "Cześć, jak się masz?"

is_same_language = checker.is_same_language(text_en, text_en_2)
print(f"Are the texts in the same language: {is_same_language}")
# Output: Are the texts in the same language: True

# Checking with a high certainty threshold for texts in different languages
is_same_language = checker.is_same_language(text_en, text_mixed, certainty=0.999)
print(f"Are the texts in the same language: {is_same_language}")
# Output:
# WARNING:root:Failed to predict language for text: Gut morning, jak are du?, returning False.
# Are the texts in the same language: False

# Checking texts in different languages
is_same_language = checker.is_same_language(text_en, text_pl, certainty=0.8)
print(f"Are the texts in the same language: {is_same_language}")
# Output: Are the texts in the same language: False
```

These examples demonstrate the main functionalities of the `LanguageChecker` class. You can use these methods to detect languages, compare texts languages, and get language predictions with confidence scores in your projects. The examples also show how the class handles cases where the language detection confidence is below the specified threshold, logging warnings when appropriate.
