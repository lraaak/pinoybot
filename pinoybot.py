"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier
Loads the trained model and vectorizer to tag new tokens.
"""

import joblib
from typing import List
from feature_utils import extract_features_for_word
import pathlib
import re
import os

print(f"I am looking for files in: {os.getcwd()}")

# ============================================================
# MODEL SELECTION
# ============================================================

# Options:
#   "lr"
#   "random_forest"
#   "svc"
MODEL_CHOICE = "random_forest"

MODEL_PATH = f"pinoybot_model_{MODEL_CHOICE}.pkl"
VEC_PATH = "pinoybot_vectorizer.pkl"

# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

script_dir = pathlib.Path(__file__).parent

model_file = script_dir / MODEL_PATH
vectorizer_file = script_dir / VEC_PATH

print(f"Loading model: {model_file}")
clf = joblib.load(model_file)

print(f"Loading vectorizer: {vectorizer_file}")
vec = joblib.load(vectorizer_file)


# ============================================================
# SPECIAL WORD HANDLING
# ============================================================

def decade_to_word(decade):
    decade = decade.lower()
    if len(decade) < 3:
        return decade

    if (decade[-1] == 's' and decade[-2].isdigit()) or \
       (decade[-2] == "'" and decade[-1] == 's' and decade[-3].isdigit()):

        decade_str = decade.replace("'", "").replace('s', '')
        if not decade_str.isdigit():
            return decade

        num = int(decade_str) % 100

        number_words = {
            0: 'hundreds', 10: 'tens', 20: 'twenties', 30: 'thirties',
            40: 'forties', 50: 'fifties', 60: 'sixties', 70: 'seventies',
            80: 'eighties', 90: 'nineties'
        }
        return number_words.get(num, decade)

    return decade


# ============================================================
# TAGGING FUNCTION
# ============================================================

def tag_language(tokens: List[str]) -> List[str]:
    token_copy = [decade_to_word(word) for word in tokens]
    features = [extract_features_for_word(word) for word in token_copy]
    X_new = vec.transform(features)
    predicted = clf.predict(X_new)
    return [str(tag) for tag in predicted]


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    sentence = "Tara! nag-coffee tayo sa Starbucks, after ng project natin. Grabe, ang saya! I love it! gustong-gusto ko na mag-haircut eh. Sige, see you soon!"

    punctuation_to_separate = r'([.,;:\"?!()])'
    tokens = re.split(r'\s+|' + punctuation_to_separate, sentence)
    tokens = [t for t in tokens if t and t.strip()]

    predicted_tags = tag_language(tokens)

    print("TAG | TOKEN")
    for tag, token in zip(predicted_tags, tokens):
        print(f"{tag} | {token}")
