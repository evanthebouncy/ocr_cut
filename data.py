import keras
from keras.datasets import mnist
import random
import numpy as np
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def make_ocr_data(seq_len):
    rid = np.random.randint(0, len(y_train), seq_len)
    randX, randY = list(x_train[rid]), np.array(list(y_train[rid])+[10])
    seqX = np.concatenate(randX, axis=1)
    return seqX, randY

def make_ocr_data_batch(seq_len, batch_n):
    xy = [make_ocr_data(seq_len) for i in range(batch_n)]
    return np.array([x for x,_ in xy]), np.array([y for _,y in xy])

def plot(seqX):
    plt.imshow(seqX)
    plt.savefig("hi.png")
    plt.close()

if __name__ == '__main__':
    print ("hello")
    seqX, Y = make_ocr_data(10)
    plot(seqX)
    print (seqX.shape, Y.shape)
    print (Y)

    seqX_batch, Y_batch = make_ocr_data_batch(5, 10)
    print (seqX_batch.shape)
    print (Y_batch.shape)
