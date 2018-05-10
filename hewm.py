# @Time    : 18-5-10 下午2:44
# @Author  : Chao
# @Email   : sttide@outlook.com
# @File    : hewm.py
# @Software: PyCharm
import tensorflow as tf
import new_reader
import time
import cv2
import numpy as np
from skimage import io
from os import *
from os.path import join
from sklearn.utils import shuffle

#超参数
batch_size = 200
image_size = 32
iters = 2000000# * 50000
image_flat = 32*32*3
regularizer = 0.0001
model_save_path = "./models/"
model_name = "cifar10"
train_image_path = "./train"
test_image_path = "./test"
# n is small, only use list
train_files = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model_path = model_save_path + model_name
if not path.exists(model_path):
    makedirs(model_path)



images, labels = new_reader.load_train()
images = images.astype(np.float32)
# very important
images = np.multiply(images, 1.0 / 255.0)
images, labels = shuffle(images, labels)

def get_weight(shape ,regularizer):
    w = tf.truncated_normal(shape, stddev=0.1)
    if regularizer != None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return tf.Variable(w)

def get_bias(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

def conv_op(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


x = tf.placeholder(tf.float32,[None,image_flat],name='x')
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 32, 32, 3])
global_steps = tf.Variable(0, trainable=False)


conv1_w = get_weight([5, 5, 3, 32],regularizer)
conv1_b = get_bias([32])
conv1 = conv_op(x_image,conv1_w)
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
pool1 = max_pool(relu1)

conv2_w = get_weight([5, 5, 32, 64], regularizer)
conv2_b = get_bias([64])
conv2 = conv_op(pool1, conv2_w)
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
pool2 = max_pool(relu2)

pool_shape = pool2.get_shape().as_list()
nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
print(nodes)
reshaped = tf.reshape(pool2,[-1,nodes])

fc1_w = get_weight([nodes,1024], regularizer)
fc1_b = get_bias([1024])
fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w)+fc1_b)
fc1_d = tf.nn.dropout(fc1, 0.5)

fc2_w = get_weight([1024,10], regularizer)
fc2_b = get_bias([10])
y = tf.nn.softmax(tf.matmul(fc1_d,fc2_w)+fc2_b,name="res")

print(y)
print(y_)
cross_entropy = -tf.reduce_sum(y_*tf.log(y+ 1e-10))
loss = cross_entropy + tf.add_n(tf.get_collection('losses'))


learning_rate = tf.train.exponential_decay(0.001, global_steps, 100, 0.99, staircase=True)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
res_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_step = tf.train.AdamOptimizer(0.0005).minimize(loss,global_step=global_steps)


saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    start_time = time.time()
    tb_time = time.time()

    for i in range(int(iters / batch_size)):
        start = (i * batch_size) % 50000
        xr = images[start:start+batch_size]
        yr = labels[start:start+batch_size]
        _, step, result,res_a= sess.run([train_step,global_steps,cross_entropy,res_accuracy],feed_dict={x:xr,y_:yr})
        if step!=0 and step%100 == 0:
            this_time = time.time()
            print("After %d steps training, using time is %.2fs! Loss is %.6f! Accuracy is %.2f%%!" % (step,this_time-start_time,result,res_a*100))
            start_time = time.time()
    print("Done!Using total time is %.2fmin!" % ( (time.time()-tb_time)/60.0) )
    saver.save(sess,model_path+'/nclass')


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
print(train_files[lab])
auccracy()

'''
data = dataset.read_train_sets(validation_size=0.16)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Validation-set:\t{}".format(len(data.valid.labels)))
'''
