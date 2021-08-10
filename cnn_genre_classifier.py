import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"

def load_data(data_path):
    # Loads training dataset from json file
    #
    #     :param data_path(str): Path to json file containing data_path
    #     :return X (ndarray): Inputs
    #     :return y (ndarray): Targets

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):
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

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 1st conv layer
        # the 1st parameter is # of filters. the 2nd is the grid size of the kernel.
        # the 3rd size is the activation function
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))

        # 1st pooling layer
        # the 1st parameter is the grid size. the second is the (horizontal, vertical) stride.
        # the 3rd is the padding. same means zero padding applied around ALL edges
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
        # batch normalization is a process that normalizes the activation in the current layer and the subsequent layer
        # it speeds up training. the models converge faster and the models are more reliable.
        # its very complicated, go read about it
    model.add(keras.layers.BatchNormalization())
    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
        # the first parameter is the number of classes that can be classified.
        # softmax creates scores for each of the 10 categories.
        # the total for all 10 neurons = 1, the probability is distributed among them
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    # create train, validation and test sets                                                   # .25 of the train set used for test set
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) # 0.2 of data for validation set

    # build the CNN net
    # X_train is a 4d array. see prepare_datasets
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train the CNN
    model.fit(X_train,
              y_train,
              validation_data=(X_validation, y_validation),
              batch_size=64,
              epochs=50)

    # evaluate the CNN on the testset
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    # make predictions on a sample