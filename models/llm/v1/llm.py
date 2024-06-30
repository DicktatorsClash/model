import os
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data.data import data

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
    for i in range(100):  # Number of iterations
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

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                         num_labels=len(label_dict))

# Tokenize and pad sequences
inputs = tokenizer(texts, max_length=128, truncation=True, padding=True, return_tensors='tf')
input_ids = inputs['input_ids'].numpy()
attention_masks = inputs['attention_mask'].numpy()

# Split data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, encoded_labels, test_size=0.2)
train_masks, val_masks, _, _ = train_test_split(attention_masks, encoded_labels, test_size=0.2)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_masks, train_labels)).shuffle(
    len(train_inputs)).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_masks, val_labels)).batch(16)

# Custom training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)


@tf.function
def train_step(input_ids, attention_masks, labels):
    with tf.GradientTape() as tape:
        outputs = distilbert_model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
    gradients = tape.gradient(loss, distilbert_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distilbert_model.trainable_variables))
    return loss


@tf.function
def val_step(input_ids, attention_masks, labels):
    outputs = distilbert_model(input_ids, attention_mask=attention_masks, labels=labels)
    loss = outputs.loss
    return loss


# Training loop
epochs = 100
best_val_loss = float('inf')
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')

    # Training
    train_loss = 0.0
    train_steps = 0
    for step, (input_ids_batch, attention_masks_batch, labels_batch) in enumerate(train_dataset):
        loss = train_step(input_ids_batch, attention_masks_batch, labels_batch)
        train_loss += tf.reduce_sum(loss)
        train_steps += input_ids_batch.shape[0]
    train_loss /= train_steps

    # Validation
    val_loss = 0.0
    val_steps = 0
    for step, (input_ids_batch, attention_masks_batch, labels_batch) in enumerate(val_dataset):
        loss = val_step(input_ids_batch, attention_masks_batch, labels_batch)
        val_loss += tf.reduce_sum(loss)
        val_steps += input_ids_batch.shape[0]
    val_loss /= val_steps

    print(f'Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        distilbert_model.save_pretrained('best_model_llm')

# Save the final model
distilbert_model.save_pretrained("model_distilbert_v2_100")


# Function to classify text and extract arguments
def classify_and_extract_arguments(text):
    # Classify text using DistilBERT
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    prediction = distilbert_model(inputs)[0]
    predicted_label = np.argmax(prediction, axis=-1)
    label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label[0])]

    # Extract custom keywords using the trained spaCy model
    doc = nlp(text)
    arguments = [ent.text for ent in doc.ents]

    return label, arguments


# Load the trained NER model for inference
nlp = spacy.load("../../../custom_ner_model")

# Example usage
print(classify_and_extract_arguments("сгенерировать ключи?"))
print(classify_and_extract_arguments("обменяй 32 токена"))
