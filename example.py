
from language_checker import LanguageChecker


checker = LanguageChecker()

text_en = "Hello, how are you?"
text_en_2 = "Hi, I'm fine. How are you?"
text_pl = "Cześć, jak się masz?"
text_mixed = "Gut morning, jak are du?"

# Simple language prediction
language_name = checker.predict_language(text_en)
print(f"Predicted language: {language_name}\n")

# Language preditction with defined minimum acceptable certainty, error if below certainty 
try: 
    language_name = checker.predict_language(text_mixed, certainty=0.999) #should raise a ValueError
except ValueError as e:
    print(f"Error: {e}\n")

# Language prediction with defined minimum acceptable certainty, no error if below certainty
try:
    language_name = checker.predict_language(text_en, certainty=0.5) # should successfuly detect language
    print(f"Predicted language: {language_name}\n")
except ValueError as e:
    raise e

# Language prediction with outputting the models prediction confidence
language_name, confidence = checker.predict_language_and_certainty(text_en)
print(f"Predicted language: {language_name} with confidence: {confidence:.2f}\n")

# Getting language candidates, with confidence scores for each
candidates = checker.predict_language_candidates(text_en, k=3)
print("Language candidates:")
for name, confidence in candidates:
    print(f"\tLanguage candidate: - {name}: {confidence:.6f}\n")

# Checking if a text is in a specific language
is_en = checker.is_language(text_en, "english")
print(f"Is text confirmed to be in English: {is_en}\n")

# Checking if a text is in a specific language with a defined minimum acceptable certainty, 
is_en = checker.is_language(text_mixed, "english", certainty=0.999) # should return False and log a warning 
print(f"Is text confirmed to in English: {is_en}\n")

# Checking if multiple texts are in the same language
is_same_language = checker.is_same_language(text_en, text_en_2) # should return True
print(f"Are the texts in the same language: {is_same_language}\n")

# Checking if multiple texts are in the same language with a defined minimum acceptable certainty
is_same_language = checker.is_same_language(text_en, text_mixed, certainty=0.999) # should retrun False and log a warning
print(f"Are the texts in the same language: {is_same_language}\n") 

# Checking if multiple texts are in the same language with a defined minimum acceptable certainty
is_same_language = checker.is_same_language(text_en, text_pl, certainty=0.8) # should return False
print(f"Are the texts in the same language: {is_same_language}\n") 





