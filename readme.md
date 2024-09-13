# Language Detection with FastText

This project provides a Python module for detecting the language of a given text using the FastText language identification model by Meta-Facebook. It includes utilities for mapping between ISO 639-3 language codes and their corresponding language names, as well as a wrapper for easier use of the AI model.

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

