import preprocess
import cnn_genre_classifier

if __name__ == "__main__":
    # You can run this if you wish to both reprocess the dataset and classify it
    # i.e. if you change some preprocess settings. Otherwise you can just run cnn_genre_classifier
    preprocess.run()
    cnn_genre_classifier.run()