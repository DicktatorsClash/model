import os
import spacy
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the spaCy NER model
nlp = spacy.load("../../../custom_ner_model")

# Load the DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained("../../../model_distilbert")

# Label dictionary (recreate it based on your training data)
label_dict = {
    'swap tokens': 0,
    'generate key': 1,
    'send tokens': 2,
    'attack country': 3
}

# Function to classify text and extract arguments
def classify_and_extract_arguments(text):
    # Classify text using DistilBERT
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = distilbert_model(inputs)
    prediction = outputs.logits
    predicted_label = np.argmax(prediction, axis=-1)
    label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label[0])]

    # Extract custom keywords using the trained spaCy model
    doc = nlp(text)
    arguments = [ent.text for ent in doc.ents]

    return label, arguments

# Example usage
print(classify_and_extract_arguments("сгенерировать ключи?"))
print(classify_and_extract_arguments("обменяй 32 токена"))
print(classify_and_extract_arguments("отправить токены"))
