
from language_checker import LanguageChecker


if __name__ == "__main__":
    checker = LanguageChecker()
    text = "Cześć, jak się masz?"

    # Getting language candidates
    candidates = checker.predict_language_candidates(text, k=3)
    print("Language candidates:")
    for name, confidence in candidates:
        print(f"- {name}: {confidence:.2f}")

    # Using predict_language_and_certainty
    language_name, confidence = checker.predict_language_and_certainty(text)
    print(f"Predicted language: {language_name} with confidence: {confidence:.2f}")

    # Using predict_language with exception handling
    try:
        language_name = checker.predict_language(text, certainty=0.5)
        print(f"Predicted language: {language_name}")
    except ValueError as e:
        print(f"Error: {e}")
