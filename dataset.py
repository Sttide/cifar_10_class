# @Time    : 18-5-10 上午10:46
# @Author  : Chao
# @Email   : sttide@outlook.com
# @File    : dataset.py
# @Software: PyCharm

import os
import glob
import numpy as np
import cv2
from os import *
from os.path import join
from sklearn.utils import shuffle

train_files = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def load_train():
    images = []
    lables = []

    print("Reading training images")
    files_path = "./train/"
    for f in listdir(files_path):
        print(f)
        image = cv2.imread(f)
        images.append(image)
        f_class = f.split('_')[0]
        inde = train_files.index(f_class)
        f_label = np.zeros(10)
        f_label[inde] = 1
        lables.append(f_label)
    images = np.array(images)
    lables = np.array(lables)
    print(np.shape(images))
    print(np.shape(lables))
    return images, lables

def load_test():
    test_path = "./test/"
    files = test_path
    X_test = []
    X_test_id = []
    print("Reading test images!")

    num = 0
    for file_class in listdir(files):
        print(file_class)
        file_subc = join(files, file_class)
        print(file_subc)
        for f1 in listdir(file_subc):
            img = cv2.imread(f1)
            X_test.append(img)
            X_test_id.append()
        pass

class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]

def read_train_sets( validation_size=0):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels = load_train()
    print(len(images)," ",len(labels))
    images, labels = shuffle(images, labels)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
        print(validation_size)
        print(images.shape[0])
        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]

        data_sets.train = DataSet(train_images, train_labels)
        data_sets.valid = DataSet(validation_images, validation_labels)

    print(data_sets)
    return data_sets


if __name__=="__main__":
    read_train_sets(0)
