import csv
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import scipy.sparse as sp

files_to_load = {
    "marriott": "marriott_calculated_features.csv",
    "hilton": "hilton_calculated_features.csv",
    "holiday_inn": "holiday_inn_calculated_features.csv"
}

master_reviews = []
master_sentiments = []
master_word_counts = []
master_count_good = []
master_count_average = []
master_count_medium = []
master_count_worst = []

hotel_row_counts = {
    "marriott": 0,
    "hilton": 0,
    "holiday_inn": 0
}

for brand, file_path in files_to_load.items():
    print(f"Reading {brand.upper()} dataset features...")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                master_reviews.append(row['review'])
                master_sentiments.append(row['sentiments'])
                master_word_counts.append(int(row['total_word_count'] or 0))
                master_count_good.append(int(row['count_good'] or 0))
                master_count_average.append(int(row['count_average'] or 0))
                master_count_medium.append(float(row['count_medium'] or 0))
                master_count_worst.append(int(row['count_worst'] or 0))
                hotel_row_counts[brand] += 1
    except FileNotFoundError:
        print(f"ERROR: Missing file '{file_path}'")

print(f"Successfully loaded {len(master_reviews)} rows.")

def clean_text(text):
    text = str(text).encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

print(f"Successfully loaded {len(master_reviews)} rows.")

master_reviews = [clean_text(r) for r in master_reviews]

print("\nConverting reviews into TF-IDF vectors...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.9
)

X_tfidf = tfidf.fit_transform(master_reviews)

print("Combining engineered features with TF-IDF matrix...")

engineered_features = sp.csr_matrix(
    [
        master_word_counts,
        master_count_good,
        master_count_average,
        master_count_medium,
        master_count_worst
    ]
).T
X_all_features = sp.hstack([X_tfidf, engineered_features])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(master_sentiments)
print("Class distribution:")
print(label_encoder.classes_)

print("\nSplitting dataset into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_all_features,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("Training XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

print("Training Completed!")

y_pred_test = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# ===============================
# LEXICON vs ML COMPARISON
# ===============================



y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_test)

print("\nClassification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

all_predictions_encoded = xgb_model.predict(X_all_features)
predicted_sentiments_decoded = label_encoder.inverse_transform(all_predictions_encoded)

print("\nExporting classified datasets...")

current_index = 0

for brand, count in hotel_row_counts.items():
    output_filename = f"{brand}_classified.csv"
    print(f"Generating: {output_filename}")
    with open(output_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review",
            "total_word_count",
            "count_good",
            "count_average",
            "count_medium",
            "count_worst",
            "sentiment",
            "predicted_sentiment"
        ])
        for i in range(current_index, current_index + count):
            writer.writerow([
                master_reviews[i],
                master_word_counts[i],
                master_count_good[i],
                master_count_average[i],
                master_count_medium[i],
                master_count_worst[i],
                master_sentiments[i],
                predicted_sentiments_decoded[i]
            ])
    current_index += count

print("\nAll classified files generated successfully!")
print("Saving model and vectorizer...")

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
