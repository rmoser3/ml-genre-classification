"""
generates a new dataset and trains a new model
"""

from preprocess import run
from cnn_genre_classifier import train_model

if __name__ == "__main__":
    run()
    train_model()