import os
from data.data import data
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
TRAIN_DATA = data

texts = [text for text, _, _ in TRAIN_DATA]
labels = [label for _, _, label in TRAIN_DATA]

# Label Dictionary
label_dict = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label_dict[label] for label in labels]

# Load the pre-trained spaCy NER model
nlp = spacy.load("../../../custom_ner_model")

# Load the pre-trained TensorFlow classification model
classification_model = tf.keras.models.load_model("../../../model_4.h5")

# Tokenizer for text preprocessing
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')
labels = np.array(encoded_labels)

# Define the function for classification and extraction of arguments
def classify_and_extract_arguments(text):
    # Classify text using the pre-trained TensorFlow model
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=padded_sequences.shape[1])
    prediction = classification_model.predict(padded)
    predicted_label = np.argmax(prediction)
    label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]

    # Extract custom keywords using the trained spaCy model
    doc = nlp(text)
    arguments = [ent.text for ent in doc.ents]

    return label, arguments

# Example usage
print(classify_and_extract_arguments("can  u burn tokens to zero address "))
