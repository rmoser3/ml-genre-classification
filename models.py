import tensorflow.keras as keras

# The model constructed by The Sound of AI
    # https://www.youtube.com/watch?v=dOG-HxpbMSw
def cnn_VV(input_shape):
    """Generates CNN model
        :param input_shape (tuple): Shape of input set
        :return model: CNN model
    """
    # create model
    model = keras.Sequential()

    # 1st conv layer
        # the 1st parameter is # of filters.
        # the 2nd is the grid size of the kernel.
        # the 3rd is the activation function
        # the 4th is the shape of the numpy array going into the layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))

    # 1st pooling layer
        # the 1st parameter is the grid size.
        # the 2nd is the (horizontal, vertical) stride.
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

def cnn_DS(input_shape):
    """
    TODO: try it without padding in the conv layers
    """
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(8, (3, 3), strides=(1,1), activation='relu', input_shape=input_shape, padding="same"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
