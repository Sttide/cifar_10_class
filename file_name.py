from os import *
from os.path import join
import shutil
import numpy as np
image_path = "./train"
train_files = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def img_rename():
    for file_num in range(10):
        new_path = image_path + '/' + train_files[file_num] + '/'
        print(new_path)
        j = 1
        for f in listdir(new_path):
            file_name = join(new_path, f)
            new_name =  train_files[file_num] + '_' + str(j) + ".jpg"
            j = j + 1
            print(file_name,new_name)
            rename(file_name,new_name)


def img_move():
    path = getcwd()
    print(path)
    for f in listdir(path):
        x = f.split('.')[-1]
        print(x)
        if x == "jpg":
            shutil.move(f, "./train/" + f)

if __name__ == "__main__":
    print(np.nan)
    pass
