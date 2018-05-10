from os import *
#new_path = "./train/ship/"
#x = listdir(new_path)
#print(len(x))
#超参数
import tensorflow as tf
import cv2
import numpy as np
from skimage import io
from os import *
from os.path import join
from sklearn.utils import shuffle
batch_size = 200
image_size = 32
iters = 1200000# * 50000
image_flat = 32*32*3
regularizer = 0.0001
model_save_path = "./models/"
model_name = "cifar10"
train_image_path = "./train"
test_image_path = "./test"
train_files = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model_path = model_save_path + model_name
if not path.exists(model_path):
    makedirs(model_path)

attribute ={
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"
}

i = 1
while i < 1+5:
    print(i)
    i = i + 1

def predict(image_path):
    img = io.imread(image_path)
    print(image_path)
    img = img / 255.0
    imgarr = np.array(img)
    imgd = imgarr.reshape([-1, 32*32*3])

    sess = tf.Session()
    saver = tf.train.import_meta_graph('./models/cifar10/nclass.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./models/cifar10'))

    input_x = sess.graph.get_tensor_by_name('x:0')
    opt = sess.graph.get_tensor_by_name('res:0')
    ret = sess.run(opt,feed_dict={input_x:imgd})
    print(ret)
    res = np.argmax(ret, 1)

    return res[0]

def auccracy():
    num = 0
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./models/cifar10/nclass.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint('./models/cifar10'))

    input_x = sess.graph.get_tensor_by_name('x:0')
    opt = sess.graph.get_tensor_by_name('res:0')

    for i in range(10):
        new_path = test_image_path + '/' + train_files[i] +'/'
        print(new_path)
        print(train_files[i])
        for f in listdir(new_path):
            file_name = join(new_path, f)
            img = cv2.imread(file_name)
            imgarr = np.array(img)
            imgd = imgarr.reshape([-1,32*32*3])
            imgd = imgd.astype(np.float32)
            image = np.multiply(imgd, 1.0 / 255.0)

            ret = sess.run(opt, feed_dict={input_x: image})
            res = np.argmax(ret, 1)
            test_lab = res[0]
            if(test_lab == i):
                num = num + 1
    print("Total auccracy: %.2f%%!" % (num/100))



lab = predict("./test/bird/batch_1_num_25.jpg")
print(lab)
print(attribute[lab],"\n")
auccracy()
