import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from feature_utils import extract_features_for_word
import joblib
import sys
import time


sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# SETTINGS
# ============================================================
DEV_MODE = False
DATASET_PATH = "MCO2_Dataset_Cleaned.xlsx"
SAVE_FEATURE_DATA = False

# ⭐⭐ MODEL CHOICE ⭐⭐
# "lr"            = LogisticRegression
# "random_forest" = RandomForestClassifier
# "svc"           = LinearSVC
MODEL_CHOICE = "lr"


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(DATASET_PATH)
print(f"Total rows loaded: {df.shape[0]}")
df = df.dropna(subset=['word']).copy()

df['final_label'] = df['corrected_label'].fillna(df['label'])


# ============================================================
# LABEL MAPPING
# ============================================================
def map_labels(tag):
    if pd.isna(tag):
        return 'OTH'
    tag = str(tag).upper()
    if 'FIL' in tag or 'CS' in tag:
        return 'FIL'
    if 'ENG' in tag:
        return 'ENG'
    return 'OTH'

df['mapped_label'] = df['final_label'].apply(map_labels)

print("\n=== DATA PREVIEW ===")
print(df.head(20))


# ============================================================
# FEATURE EXTRACTION WITH SENTENCE CONTEXT
# ============================================================
feature_dicts = []
prev_word, prev_pred = None, None
prev_sentence = None

print("\nExtracting features...")
start_feat = time.time()

for i, row in df.iterrows():

    # Reset context at sentence boundaries
    if prev_sentence is None or row["sentence_id"] != prev_sentence:
        prev_word, prev_pred = None, None

    feats = extract_features_for_word(row['word'], prev_word, prev_pred)
    feature_dicts.append(feats)

    prev_word = row['word']
    prev_pred = row['mapped_label']
    prev_sentence = row['sentence_id']

end_feat = time.time()
print(f"Feature extraction time: {end_feat - start_feat:.2f} seconds")


# ============================================================
# VECTORIZE FEATURES
# ============================================================
vec = DictVectorizer(sparse=True)

start_vec = time.time()
X = vec.fit_transform(feature_dicts)
y = df['mapped_label'].values
end_vec = time.time()

print(f"Vectorization time: {end_vec - start_vec:.2f} seconds")
print("Feature matrix shape:", X.shape)


# ============================================================
# OPTIONAL SAVE OF FEATURE DATA
# ============================================================
if SAVE_FEATURE_DATA:
    feature_df = pd.DataFrame(feature_dicts)
    feature_df['label'] = y
    feature_df.to_excel("MCO2_Feature_Expanded.xlsx", index=False)
    print("Saved feature-expanded dataset → MCO2_Feature_Expanded.xlsx")


# ============================================================
# MODEL SELECTION
# ============================================================
def build_model(choice):
    if choice == "lr":
        print("\nUsing Logistic Regression")
        return LogisticRegression(
            max_iter=1200,
            class_weight='balanced',
            random_state=42
        )

    elif choice == "random_forest":
        print("\nUsing Random Forest Classifier")
        return RandomForestClassifier(
        n_estimators=120,
        max_depth=70,
        max_features="log2",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    elif choice == "svc":
        print("\nUsing Linear SVC")
        return LinearSVC(
            class_weight="balanced",
            random_state=42
        )

    else:
        print("\n❌ Invalid MODEL_CHOICE")
        sys.exit()


model = build_model(MODEL_CHOICE)


# ============================================================
# FINAL TRAIN–VAL–TEST SPLIT
# ============================================================
print("\nTraining model...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

print(f"Training time: {end_train - start_train:.2f} seconds")



# ============================================================
# PERFORMANCE
# ============================================================
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

print("\n=== Classification Report (Validation) ===")
print(classification_report(y_val, model.predict(X_val), digits=3))

print("\n=== Confusion Matrix (Validation) ===")
print(confusion_matrix(y_val, model.predict(X_val), labels=model.classes_))


# ============================================================
# SAVE MODEL
# ============================================================
joblib.dump(model, f"pinoybot_model_{MODEL_CHOICE}.pkl")
joblib.dump(vec, "pinoybot_vectorizer.pkl")

print(f"\nModel saved as pinoybot_model_{MODEL_CHOICE}.pkl")
print("Vectorizer saved as pinoybot_vectorizer.pkl")
