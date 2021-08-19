# Preprocess settings
DATASET_PATH = "genres/" # path to audio files
JSON_PATH = "data.json" # path to file that will contain the dataset
SAMPLE_RATE = 22050  # number of samples taken per second
DURATION = 30 # measured in seconds
N_MFCC = 13 # number of mfcc coefficients per mfcc vector
N_FFT = 2048 # fft size
HOP_LENGTH = 128 # step size
NUM_SEGMENTS = 30 # number of segments per track

# Model settings
MODEL_NAME = "cnn_V2" # for the csv file containing the metrics
LEARNING_RATE = 0.00015 # how much to change the model in response to error
BATCH_SIZE = 32 # how many training samples are used per epoch
EPOCHS = 65 # for each epoch, the model trains on some number of training samples
                # and is tested on the validation set
