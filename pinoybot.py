"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""
import joblib
from typing import List
from feature_utils import extract_features_for_word

"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier
Loads the trained Decision Tree model and vectorizer to tag new tokens.
"""

MODEL_PATH = "pinoybot_model.pkl"
VEC_PATH = "pinoybot_vectorizer.pkl"

clf = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)

def tag_language(tokens: List[str]) -> List[str]:
    """Tag each token as FIL, ENG, or OTH."""
    features = [extract_features_for_word(word) for word in tokens]
    X_new = vec.transform(features)
    predicted = clf.predict(X_new)
    return [str(tag) for tag in predicted]

if __name__ == "__main__":
    example_tokens =[
    "chinat", "niya", "ako", "kanina", ".", "magshopping", "ako"
    ]

    print("Tokens:", example_tokens)
    predicted_tags = tag_language(example_tokens)
    
    print("TAG | TOKEN")
    for i in range(len(example_tokens)):
        token = example_tokens[i]
        tag = predicted_tags[i]
        print(f"{tag} | {token}")

