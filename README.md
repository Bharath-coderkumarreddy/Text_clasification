# Text_clasification

entiment Analysis using TensorFlow and Pre-trained Embeddings
This project demonstrates how to perform sentiment analysis on the IMDB Reviews dataset using TensorFlow and pre-trained word embeddings from Google News (via TensorFlow Hub). The goal of this implementation is to predict whether a given movie review is positive or negative.

Requirements:-
TensorFlow
TensorFlow Hub
TensorFlow Datasets
NumPy
You can install these dependencies using pip:

pip install tensorflow tensorflow-hub tensorflow-datasets numpy

Code Overview
The provided Python code accomplishes the following steps:

1. Import Necessary Libraries
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tdfs
import tensorflow as tf


These libraries are required for loading datasets, implementing the model, and working with pre-trained embeddings.

3. Load IMDB Dataset
The IMDB Reviews dataset is loaded from TensorFlow Datasets (TFDS) with a split of training, validation, and test sets.


train_data, val_data, test_data = tdfs.load(
    name="imdb_reviews", split=('train', 'test[:40%]', 'test[40%:]'), as_supervised=True
)
train_data: 60% of the data for training.
val_data: 20% of the data for validation.
test_data: 20% of the data for testing.
3. Inspect Sample Data
A batch of 5 training examples and labels is fetched to inspect the format of the data.


train_example_batch, train_labels_batch = next(iter(train_data.batch(5)))
train_example_batch
train_labels_batch
4. Load Pre-trained Word Embeddings
The pre-trained word embeddings from Google News are loaded using TensorFlow Hub.

hub_layer = hub.KerasLayer("https://kaggle.com/models/google/gnews-swivel/frameworks/TensorFlow2/variations/tf2-preview-20dim/versions/1", 
                           output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)
This layer maps words in the input text to a 20-dimensional vector. The embeddings are trainable, meaning they will be fine-tuned during training.

5. Define the Model Architecture
A simple feedforward neural network is defined using Keras' Sequential API.


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1,))
The hub_layer provides the pre-trained embeddings.
A Dense layer with ReLU activation is used to add complexity.
The final layer outputs a single value (logit) indicating sentiment (positive/negative).
6. Compile the Model
The model is compiled with the Adam optimizer and Binary Cross-Entropy loss function.

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
BinaryCrossentropy is used as the loss function since the problem is binary (positive/negative sentiment).
from_logits=True indicates that the model outputs raw logits (before applying a sigmoid).
7. Train the Model
The model is trained for 5 epochs with shuffled training data in batches of 100 samples. The validation data is provided for evaluating performance during training.


history = model.fit(train_data.shuffle(10000).batch(100), epochs=5, validation_data=val_data.batch(100), verbose=1)
8. Evaluate the Model
After training, the model is evaluated on the test data to compute the final performance metrics (accuracy and loss).

results = model.evaluate(test_data.batch(100), verbose=2)
Output
The model will print the summary of the architecture, training progress, and final evaluation metrics (loss and accuracy) after the training is complete.

Notes
The dataset used is the IMDB Reviews, a common dataset for binary sentiment classification.
The pre-trained embeddings used here are from Google News and are fine-tuned during training.
You can adjust the batch size, number of epochs, and model architecture for better results.
