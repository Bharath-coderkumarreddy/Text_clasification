import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# Load IMDB dataset
train_data, val_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train', 'test[:40%]', 'test[40%:]'),
    as_supervised=True
)

# Pre-trained text embedding layer
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
    output_shape=[20],
    input_shape=[],
    dtype=tf.string,
    trainable=True
)

# Define Sentiment Analysis Model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Show model structure
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_data.shuffle(10000).batch(100),
    epochs=5,
    validation_data=val_data.batch(100),
    verbose=1
)

# Evaluate model
results = model.evaluate(test_data.batch(100), verbose=2)
print("Test Results:", results)

# Plot training history
epochs = range(1, 6)
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model.save('models/sentiment_model')

# Example prediction function
def predict_review(review):
    prediction = model.predict([review])[0][0]
    sentiment = "Positive Review" if prediction > 0 else "Negative Review"
    return sentiment

# Test example
example_review = "This movie is fantastic!"
print("Prediction:", predict_review(example_review))
