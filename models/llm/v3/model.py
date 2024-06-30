import os
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from data.data import data

# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# Use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the dataset
TRAIN_DATA = data

texts = [text for text, _, _ in TRAIN_DATA]
labels = [label for _, _, label in TRAIN_DATA]

# Label Dictionary
label_dict = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label_dict[label] for label in labels]

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_dict)
)

# Tokenize and pad sequences
inputs = tokenizer(texts, max_length=128, truncation=True, padding=True, return_tensors='np')
input_ids = inputs['input_ids']
attention_masks = inputs['attention_mask']

# Split data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, encoded_labels, test_size=0.2, random_state=42
)
train_masks, val_masks = train_test_split(
    attention_masks, test_size=0.2, random_state=42
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((train_inputs, train_masks), train_labels)).shuffle(len(train_inputs)).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_inputs, val_masks), val_labels)).batch(16)

# Compile model
distilbert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = distilbert_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the best model
distilbert_model.save_pretrained("best_model_llm_v5")

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    outputs = distilbert_model(inputs)
    prediction = outputs.logits
    predicted_label = np.argmax(prediction, axis=-1)
    label = list(label_dict.keys())[list(label_dict.values()).index(predicted_label[0])]
    return label

# Example usage
print(classify_text("сгенерировать ключи?"))
print(classify_text("обменяй 32 токена"))
print(classify_text("отправить токены в страну"))
print(classify_text("атаковать страну"))
