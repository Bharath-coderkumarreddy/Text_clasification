import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tdfs
import tensorflow as tf


train_data, val_data, test_data = tdfs.load(name="imdb_reviews", split=('train', 'test[:40%]', 'test[40%:]'), as_supervised=True)
train_example_batch, train_labels_batch = next(iter(train_data.batch(5)))
train_example_batch
train_labels_batch
hub_layer = hub.KerasLayer("https://kaggle.com/models/google/gnews-swivel/frameworks/TensorFlow2/variations/tf2-preview-20dim/versions/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_example_batch[:3])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1,))

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(100), epochs = 5, validation_data=val_data.batch(100),verbose=1)

results = model.evaluate(test_data.batch(100), verbose=2)
results
