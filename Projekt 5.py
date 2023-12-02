from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# Function to create a model with a specified number of convolutional layers
def create_model(num_conv_layers):
    model = Sequential()

    for _ in range(num_conv_layers):
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Split data into training and testing sets (70% training, 30% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

# Select two out of ten classes
selected_classes = [3, 5]

# Map original labels to the range [0, 1]
y_train_mapped = np.where(np.isin(y_train, selected_classes), 1, 0)
y_val_mapped = np.where(np.isin(y_val, selected_classes), 1, 0)
y_test_mapped = np.where(np.isin(y_test, selected_classes), 1, 0)

# Convert labels to binary form
num_classes = len(selected_classes)
y_train = to_categorical(y_train_mapped, num_classes)
y_val = to_categorical(y_val_mapped, num_classes)
y_test = to_categorical(y_test_mapped, num_classes)

# Number of iterations (train the classifier 100 times)
num_iterations = 1

# Try classifiers with different numbers of convolutional layers
for num_conv_layers in [1, 2, 3]:
    print(f"Classifier with {num_conv_layers} convolutional layers:")

    for iteration in range(1, num_iterations + 1):
        print(f"Iteration {iteration}/{num_iterations}")

        # Create a classifier with a specified number of convolutional layers
        model = create_model(num_conv_layers)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), verbose=1)

        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Accuracy on test data: {test_accuracy * 100:.2f}%\n")

    # Print the model architecture
    model.summary()
    print("\n" + "="*50 + "\n")