"""
generates a new dataset guided by the parameters in preprocess.py, then
"""

from preprocess import run
from cnn_genre_classifier import train_model

if __name__ == "__main__":
    run()
    train_model()