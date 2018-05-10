from os import *
from os.path import join
import numpy as np
import time
import cv2

train_files = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#img = io.imread("./train/ship/batch_1_num_8.jpg")
#print(img.shape)

image_path = "./train"


def load_train():
    images = []
    lables = []

    print("Reading training images")
    files_path = "./train/"
    for f in listdir(files_path):
        file_name = join(files_path,f)
        image = cv2.imread(file_name)
        images.append(image)
        #print(image)
        f_class = f.split('_')[0]
        inde = train_files.index(f_class)
        f_label = np.zeros(10)
        f_label[inde] = 1
        lables.append(f_label)
    images = np.array(images)
    lables = np.array(lables)
    images = images.reshape([50000,32*32*3])
    lables = lables.reshape([-1, 10])
    print(np.shape(images))
    print(np.shape(lables))
    print("Training data have been complished!")
    return images, lables



if __name__ == "__main__":
    load_train()
    start_time = time.time()
    this_time = time.time()
    step = 1
    t = start_time - this_time
    print("After %d steps training, using time is %f" % (step,t) )
