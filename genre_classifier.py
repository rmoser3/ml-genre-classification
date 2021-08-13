import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

"""
This classifier is a DNN, not a CNN. It is not as accurate as the CNN.
It is currently kept for reference and may be deleted when necessary. 
"""

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert mfcc lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split data into train and test
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # build the network architecture
    # Suggested - 5 - Computing in neural networks
    model = keras.Sequential([
        # input layer
        # Flatten takes a multidimensional array and converts it to 1d
        # For each track, we have many mfcc vectors. Each mfcc vector is taken at a specific interval (hop length)
        # The first dimension is the intervals, the second dimension is the mfcc values for that interval

        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        # Rectified Linear Unit (ReLU) - ReLU(h) = 0 if h < 0, h if >= 0
        # With ReLU, better convergence and reduced likelihood of vanishing gradient (we can train the network way faster)
        # Vanishing gradient?? Learn about it.
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        # For the output layer, the first parameter is the number of classes we have (10 genres)
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # train network

    # EPOCHS - for a single epoch, the model trains on batch_size number of samples
    # BATCHING:
        # Batch size - how many training samples we are passing over per epoch
        # Stochastic - 1 sample at a time. Quick, but inaccurate
        # Full batch - the whole training set. Slow, memory intensive, accurate
        # Mini-batch - subset of the data set. Best of the two worlds.

    history = model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=100,
              batch_size=300)

    # plot accuracy and error over epochs

    plot_history(history)

    # the issue of overfitting to increase accuracy. https://www.youtube.com/watch?v=Gf5DO6br0ts
    # next, upgrade to CNN https://www.youtube.com/watch?v=t3qWfUYJEYU
                         # https://www.youtube.com/watch?v=dOG-HxpbMSw

