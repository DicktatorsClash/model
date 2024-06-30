import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from data.data import data

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from data.data import data


label_dict = {
    'swap tokens': 0,
    'generate key': 1,
    'send tokens': 2,
    'bomb country': 3
}

def train_model():
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

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(train_vectors, train_labels)

    # Evaluate the model
    val_predictions = model.predict(val_vectors)
    print(classification_report(val_labels, val_predictions))

    # Save the model and vectorizer
    joblib.dump(model, 'models/classic/v1/logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'models/classic/v1/tfidf_vectorizer.pkl')

def test_model(text):
    # Load the model and vectorizer
    model = joblib.load('models/classic/v1/logistic_regression_model.pkl')
    vectorizer = joblib.load('models/classic/v1/tfidf_vectorizer.pkl')

    # Transform the text and make a prediction
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return label_dict[prediction[0]]



