import os
import librosa
import math
import json

DATASET_PATH = "/Users/ryan.moser/Downloads/genres"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050  # number of samples taken per second
DURATION = 30 # measured in seconds
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
NUM_SEGMENTS = 20

SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_NUM_MFCC_VECTORS_PER_SEGMENT = math.ceil(NUM_SAMPLES_PER_SEGMENT / HOP_LENGTH)

def save_mfcc(dataset_path, json_path, n_mfcc, n_fft, hop_length, num_segments):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
            :param dataset_path (str): Path to dataset
            :param json_path (str): Path to json file used to save MFCCs
            :param n_mfcc (int): Number of coefficients to extract
            :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
            :param hop_length (int): Sliding window for FFT. Measured in # of samples
            :param: num_segments (int): Number of segments we want to divide sample tracks into
            :return:
    """

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # loop through all the genres
    # dirpath - the folder we're currently in. dirnames - the folders inside dirpath. filenames - the files inside dirnames
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/") # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(NUM_SEGMENTS):
                    start_sample = NUM_SAMPLES_PER_SEGMENT * s # s = 0 -> 0
                    finish_sample = start_sample + NUM_SAMPLES_PER_SEGMENT # s = 0 -> num_samples_per_second

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == EXPECTED_NUM_MFCC_VECTORS_PER_SEGMENT:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))
    print("MFCC: "+str(len(data["mfcc"])))
    print("Labels: "+str(len(data["labels"])))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def run():
    save_mfcc(DATASET_PATH, JSON_PATH, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS)
