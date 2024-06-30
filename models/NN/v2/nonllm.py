import os
import spacy
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Embedding, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import tensorflow as tf
import numpy as np
from data.data import data
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the dataset
TRAIN_DATA = data

texts = [text for text, _, _ in TRAIN_DATA]
labels = [label for _, _, label in TRAIN_DATA]

# Label Dictionary
label_dict = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label_dict[label] for label in labels]

# Function to validate and adjust entity offsets
def validate_and_adjust_entities(nlp, text, entities):
    doc = nlp.make_doc(text)
    valid_entities = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
    return valid_entities

# Function to remove overlapping entities
def remove_overlapping_entities(annotations):
    entities = sorted(annotations["entities"], key=lambda x: (x[0], x[1]))
    non_overlapping_entities = []
    prev_end = -1
    for start, end, label in entities:
        if start >= prev_end:
            non_overlapping_entities.append((start, end, label))
            prev_end = end
    annotations["entities"] = non_overlapping_entities

# Load the spaCy model for validation
nlp = spacy.blank("en")

# Clean and validate the dataset
for item in TRAIN_DATA:
    text, annotations, _ = item
    remove_overlapping_entities(annotations)
    annotations["entities"] = validate_and_adjust_entities(nlp, text, annotations["entities"])

# Load the spaCy model for training
nlp = spacy.load("en_core_web_sm")

# Add custom NER component if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add custom labels to the NER component
for _, annotations, _ in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipes during training to only train NER
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for i in range(100):  # Reduced iterations to avoid overfitting
        print(f"Iteration {i + 1}")
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for text, annotations, _ in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses, drop=0.5)
        print(losses)

# Save the trained NER model
nlp.to_disk("custom_ner_model")

# Tokenize Text and Pad Sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')
labels = np.array(encoded_labels)

# Callbacks for early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('../../../best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# Updated Model with GRU and more complexity
embedding_dim = 256
model = tf.keras.Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=padded_sequences.shape[1], trainable=True),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(GRU(64)),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(len(label_dict), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    padded_sequences,
    labels,
    epochs=100,  # Reduced epochs for faster training
    batch_size=256,
    validation_split=0.2,
    callbacks=[reduce_lr, model_checkpoint]
)

# Save the trained classification model
model.save("model_7.h5")

# Function to classify text and extract arguments
def classify_and_extract_arguments(text):
    # Classify text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=padded_sequences.shape[1])
    prediction = model.predict(padded)
    predicted_label = np.argmax(prediction)
    label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]

    # Extract custom keywords using the trained spaCy model
    doc = nlp(text)
    arguments = [ent.text for ent in doc.ents]

    return label, arguments

# Load the trained NER model for inference
nlp = spacy.load("../../../custom_ner_model")

# Example usage
print(classify_and_extract_arguments("сгенерировать ключи?"))
print(classify_and_extract_arguments("обменяй 32 токена"))
