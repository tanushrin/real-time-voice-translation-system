import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, load_pre_trained_model
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def set_nontrainable_layers(model):
    model.trainable = False
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(256, activation='relu')
    dropout_layer_1 = layers.Dropout(0.3)
    dense_layer_2 = layers.Dense(128, activation='relu')
    prediction_layer = layers.Dense(1, activation='sigmoid')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dropout_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    return model

def build_model():
    model = load_pre_trained_model()
    model = add_last_layers(model)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# Load the dataset
X, y = load_data()
data = split_data(X, y, test_size=0.1, valid_size=0.1)

# Construct the model
model = build_model()

tensorboard = TensorBoard(log_dir="logs")
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 32
epochs = 300

model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size,
          validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])

model.save("../raw_data/results/model.h5")

# Evaluate the model using the testing set
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")



