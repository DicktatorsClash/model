import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data.data import data

# Load the dataset
TRAIN_DATA = data

texts = [text for text, _, _ in TRAIN_DATA]
labels = [label for _, _, label in TRAIN_DATA]

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
val_vectors = vectorizer.transform(val_texts)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_vectors, train_labels)

# Evaluate the model
val_predictions = model.predict(val_vectors)
print(classification_report(val_labels, val_predictions))

# Function to classify text
def classify_text(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

# Example usage
print(classify_text("сгенерировать ключи?"))
print(classify_text("обменяй 32 токена"))
print(classify_text("отправить токены в страну"))
print(classify_text("атаковать страну"))
