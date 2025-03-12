import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

data_dir = "data"
models_dir = "models"
doodles_list_file = "doodles_to_train.txt"

os.makedirs(models_dir, exist_ok=True)

with open(doodles_list_file, "r") as f:
    doodles_to_train = [line.strip() for line in f.readlines()]

def preprocess_doodle(drawing, max_strokes=100):
    strokes = np.zeros((max_strokes, 3))
    time_step = 0
    idx = 0
    
    for stroke in drawing:
        for i in range(len(stroke[0])):
            if idx >= max_strokes:
                break
            strokes[idx] = [stroke[0][i], stroke[1][i], time_step]
            time_step += 1
            idx += 1
    return strokes

X, y = [], []
labels = {}

for label_idx, doodle in enumerate(doodles_to_train):
    file_path = os.path.join(data_dir, f"{doodle}.ndjson")
    labels[doodle] = label_idx
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            X.append(preprocess_doodle(data["drawing"]))
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

X[:, :, :2] /= np.max(X[:, :, :2])

def split_data(X, y, percentage=0.2):
    data_size = len(X)
    split_index = int(data_size * percentage)
    
    indices = list(range(data_size))
    random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]
    
    return X_train, X_test, y_train, y_test

train_percentage = 0.8

X_train, X_test, y_train, y_test = split_data(X, y, train_percentage)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    layers.LSTM(256, return_sequences=True, input_shape=(100, 3), kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.LSTM(128, kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(len(doodles_to_train), activation="softmax")
])

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(train_dataset, epochs=20, batch_size=64, validation_data=test_dataset, callbacks=[early_stopping])

model_path = os.path.join(models_dir, "doodle_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")