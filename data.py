import keras
from keras.datasets import mnist
import random
import numpy as np
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def crop_1digit(x):
    pad = np.zeros((28,14))
    x = np.concatenate((pad,x,pad),axis=1)
    to_crop_left = random.randint(0, 19)
    to_crop_right = 56 - random.randint(0, 19)
    x_crop = x[:, to_crop_left : to_crop_right]
    return x_crop


def make_ocr_data(seq_len):
    full_length = seq_len * 28
    rid = np.random.randint(0, len(y_train), seq_len)
    randX, randY = list(x_train[rid]), sum(list(y_train[rid])) % 10

    croppedX = []
    for X in randX:
        croppedX.append(crop_1digit(X))
    seqX = np.concatenate(croppedX, axis=1)

    return seqX, randY

# batch of 1 for now
def make_ocr_data_batch(seq_len):
    xy = [make_ocr_data(seq_len) for i in range(1)]
    return np.array([x for x,_ in xy]), np.array([y for _,y in xy])

def plot(seqX,name='hi'):
    plt.imshow(seqX)
    plt.savefig(f"drawings/{name}.png")
    plt.close()

if __name__ == '__main__':
    print ("hello")
    seq_len = np.random.randint(1, 6)
    seqX, Y = make_ocr_data(seq_len)
    plot(seqX)
    print (seqX.shape, Y.shape)
    print (Y)

    # seqX_batch, Y_batch = make_ocr_data_batch(2, 40)
    # print (seqX_batch.shape)
    # print (Y_batch.shape)
