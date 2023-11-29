import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, load_pre_trained_model

# Load the dataset
X, y = load_data()

# Split the data into training, validation, and testing sets
data = split_data(X, y, test_size=0.1, valid_size=0.1)

y_train_list = data['y_train'].reshape(-1).tolist()  # Convert the NumPy array to a list

# Compute class weights using the list version of y_train
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_list),
    y=y_train_list
)

class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Construct the model
model = load_pre_trained_model()

# Use TensorBoard to view metrics
tensorboard = TensorBoard(log_dir="logs")

# Define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

batch_size = 64
epochs = 100

# Train the model using the training set and validate using the validation set
model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size,
          validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping],
          class_weight=class_weights_dict)  # Pass class weights here

# Save the model to a file
model.save("../raw_data/results/model.h5")

# Evaluate the model using the testing set
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

