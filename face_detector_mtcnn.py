from glob import glob

import cv2
import numpy as np
import tensorflow as tf
import detect_face

from keras.utils import np_utils
from sklearn.datasets import load_files


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def main():

    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, "./saved_models/")

    minsize = 40  # minimum size of face
    threshold = [0.6, 0.7, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    human_files = np.array(glob("/home/slavo/Dev/face-image-recognition/lfw/*/*"))
    human_files_short = human_files[:400]

    model_predictions = []
    for human_file in human_files_short:
        draw = cv2.imread(human_file)
        img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        model_prediction = bounding_boxes.shape[0]
        print('Total %d face(s) detected' % model_prediction)
        model_predictions.append(model_prediction)

    # Report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) > 0) / len(model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)


if __name__ == '__main__':
    main()
