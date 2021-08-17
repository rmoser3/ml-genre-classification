import os
import csv
import cnn_genre_classifier as cnn
import preprocess as pp
from pathlib import Path

def run():
    # field names
    fields = ['MODEL_NAME', 'N_MFCC', 'N_FFT', 'HOP_LENGTH', 'NUM_SEGMENTS', 'LEARNING_RATE', 'BATCH_SIZE', 'EPOCHS', 'ACCURACY']

    # data rows of csv file
    row = [[cnn.MODEL_NAME, str(pp.N_MFCC), str(pp.N_FFT), str(pp.HOP_LENGTH), str(pp.NUM_SEGMENTS), str(cnn.LEARNING_RATE), str(cnn.BATCH_SIZE), str(cnn.EPOCHS), str(cnn.ACCURACY)]]


    # name of csv file
    filename = cnn.MODEL_NAME + ".csv"

    # writing to an existing csv file
    if(is_file()):
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(row)
    # writing to a new csv file
    else:
        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the fields
            csvwriter.writerow(fields)
            # writing the data rows
            csvwriter.writerows(row)


def is_file():
    file = Path(cnn.MODEL_NAME + ".csv")
    if(os.path.isfile(file)):
        return True
    else:
        return False


if __name__ == "__main__":
    run()
