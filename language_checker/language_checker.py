import fasttext
import csv
import requests
import os
import logging as log
from typing import Dict, List, Tuple, Optional
from huggingface_hub import hf_hub_download


class LanguageCodes:
    """
    Handles mapping between ISO 639-3 language codes and language names.

    Uses ISO 639-3 language codes from https://www.iso.org/iso-639-language-code.
    Downloads and parses the TSV file with language codes from:
    https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab

    The TSV file contains the following columns:
    Id, Part2b, Part2t, Part1, Scope, Language_Type, Ref_Name, Comment.

    This class uses only 'Id' and 'Ref_Name' columns to build a mapping between language codes and names.
    """

    def __init__(self):
        """Initializes the LanguageCodes class by downloading and parsing the ISO 639-3 language codes TSV file."""
        self.codes: Dict[str, str] = {}
        self._iso_file_name = "iso-639-3.tab"
        # Check if file exists
        if not os.path.exists(self._iso_file_name):
            url = "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab"
            r = requests.get(url)
            with open(self._iso_file_name, 'wb') as f:
                f.write(r.content)
        with open(self._iso_file_name, encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                self.codes[row[0]] = row[6]

    def get_language_name(self, lang_code: str) -> str:
        """
        Returns the language name corresponding to a given ISO 639-3 language code.

        Args:
            lang_code (str): The ISO 639-3 language code.

        Returns:
            str: The language name if found; otherwise, "Unknown".
        """
        return self.codes.get(lang_code, "Unknown")

    def get_language_code(self, language_name: str) -> str:
        """
        Returns the ISO 639-3 language code corresponding to a given language name.

        Args:
            language_name (str): The language name.

        Returns:
            str: The language code if found; otherwise, "Unknown".
        """
        for code, lang in self.codes.items():
            if lang == language_name:
                return code
        return "Unknown"


class LanguageChecker:
    """
    Detects the language of a given text using the FastText language identification model by Meta-Facebook.

    It uses the FastText model to predict the language of the text and provides methods to get the language name
    and possible language candidates.
    """

    def __init__(self):
        """Initializes the LanguageChecker by downloading and loading the FastText language identification model."""
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.codes = LanguageCodes()

    def _predict(self, text: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """
        Predicts the top k language labels and their probabilities for the given text.

        Args:
            text (str): The input text to analyze.
            k (int, optional): The number of top predictions to return. Defaults to 3.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of language labels and a list of corresponding probabilities.
        """
        return self.model.predict(text, k=k)

    def _language_code_from_label(self, language_label: str) -> str:
        """
        Extracts the ISO 639-3 language code from a FastText language label.

        Labels follow the format '__label__<language_code>_<alphabet_type>'.

        Args:
            language_label (str): The FastText language label.

        Returns:
            str: The extracted language code.
        """
        lang_code_with_alphabet = language_label.split("__")[-1]
        lang_code = lang_code_with_alphabet.split("_")[0]
        return lang_code

    def _language_name_from_label(self, language_label: str) -> str:
        """
        Converts a FastText language label to a human-readable language name.

        Args:
            language_label (str): The FastText language label.

        Returns:
            str: The corresponding language name.
        """
        language_code = self._language_code_from_label(language_label)
        return self.codes.get_language_name(language_code)

    def predict_language(self, text: str, certainty: Optional[float] = None) -> str:
        """
        Predicts the language of the given text with an optional certainty threshold.

        Args:
            text (str): The input text to analyze.
            certainty (float, optional): The confidence threshold between 0 and 1. If set, the method will raise an exception if the prediction confidence is below this threshold.

        Returns:
            str: The predicted language name if confidence exceeds the threshold; otherwise, raises an exception.

        Raises:
            ValueError: If the prediction confidence is below the specified certainty threshold.
        """
        language_labels, confidences = self._predict(text)
        language_label = language_labels[0]
        confidence = confidences[0]
        language_name = self._language_name_from_label(language_label)
        if certainty is None or confidence > certainty:
            return language_name
        else:
            raise ValueError(f"Language detection confidence {confidence:.2f} is below the threshold of {certainty:.2f}.")

    def predict_language_candidates(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns the top k language candidates and their confidences for the given text.

        Args:
            text (str): The input text to analyze.
            k (int, optional): The number of top candidates to return. Defaults to 3.

        Returns:
            List[Tuple[str, float]]: A list where each tuple contains a language name and its confidence score.
        """
        language_labels, confidences = self._predict(text, k=k)
        language_candidates: List[Tuple[str, float]] = []
        for i in range(k):
            language_label = language_labels[i]
            confidence = confidences[i]
            language_name = self._language_name_from_label(language_label)
            language_candidates.append((language_name, confidence))
        return language_candidates

    def predict_language_and_certainty(self, text: str) -> Tuple[str, float]:
        """
        Predicts the language of the given text and returns both the language name and the prediction confidence.

        Args:
            text (str): The input text to analyze.

        Returns:
            Tuple[str, float]: A tuple containing the predicted language name and the prediction confidence score.
        """
        language_labels, confidences = self._predict(text)
        language_label = language_labels[0]
        confidence = confidences[0]
        language_name = self._language_name_from_label(language_label)
        return language_name, confidence

    def is_same_language(self, *texts: str, certainty: Optional[float] = None) -> bool:
        """
        Checks if all the given texts are written in the same language.

        Args:
            *texts (str): The input texts to compare.
            certainty (float, optional): The confidence threshold between 0 and 1.

        Returns:
            bool: True if all texts are in the same language; otherwise, False.
        """
        languages = set()
        for text in texts:
            try:
                language = self.predict_language(text, certainty=certainty)
                languages.add(language)
            except ValueError:
                # If the certainty threshold is not met for any text, return False
                log.warning(f"Failed to predict language for text: {text}, returning False.")
                return False
        return len(languages) == 1

    def is_language(self, language_name: str, text: str, certainty: Optional[float] = None) -> bool:
        """
        Checks if the given text is written in the specified language.

        Args:
            language_name (str): The language name to check.
            text (str): The input text to analyze.
            certainty (float, optional): The confidence threshold between 0 and 1.

        Returns:
            bool: True if the text is written in the specified language; otherwise, False.
        """
        try:
            predicted_language = self.predict_language(text, certainty=certainty)
            return predicted_language.lower() == language_name.lower()
        except ValueError:
            log.warning(f"Failed to predict language for text: {text}, returning False.")
            return False
