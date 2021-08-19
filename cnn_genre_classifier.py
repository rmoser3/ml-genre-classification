import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import models
import random
import csv_utility
from settings import JSON_PATH as DATA_PATH, MODEL_NAME, LEARNING_RATE, \
    BATCH_SIZE, EPOCHS

def load_data(data_path):
    """
        Loads training dataset from json file

        :param data_path(str): Path to json file containing data_path
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """


    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"]) # Shape of X = [num_tracks * num_segments, expected_num_mfcc_vectors_per_segment, n_mfcc]
    y = np.array(data["labels"]) # Shape of Y = [num_tracks * num_segments]
    return X, y

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
        :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
        :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
        :return X_train (ndarray): Input training set
        :return X_validation (ndarray): Input validation set
        :return X_test (ndarray): Input test set
        :return y_train (ndarray): Target training set
        :return y_validation (ndarray): Target validation set
        :return y_test (ndarray): Target test set
    """
    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (130, 13, 1), 130 is the number of time bins, 13 MFCC values per time bin, 1 is the depth
    # the 3 dots = give me what i have so far, np.newaxis means add a new axis on top of that
    X_train = X_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test



def predict(model, X, y):
    """Predict a single sample using the trained model
        :param model: Trained classifier
        :param X: Input data
        :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]

    # perform prediction
    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

def load_model(model_path):
    """
    Load a saved model and test it on the test set
    """

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    model = keras.models.load_model(model_path)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on test set is: {}".format(test_accuracy))

def train_model():
    """
    Train a new model and save it
    TODO: Get probability distribution on test set to pass to score to vote function
    """
    # create train, validation and test sets                                                   # 0.25 of the train set used for test set
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) # 0.2 of data for validation set

    # build the CNN net
    # X_train is a 4d array. see prepare_datasets
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    if "cnn_V1" in MODEL_NAME:
        model = models.cnn_V1(input_shape=input_shape)
    elif "cnn_V2" in MODEL_NAME:
        model = models.cnn_V2(input_shape=input_shape)
    else:
        print("Error: model not found. The model name string should correspond" \
              "with one of the model names in models.py.")
        return

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()

    # train the CNN
    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_validation, y_validation),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate the CNN on the testset
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on test set is: {}".format(test_accuracy))

    csv_utility.run(test_accuracy)

    # make predictions on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)

    model.save(MODEL_NAME+'/')


