import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
e = np.e


def plot_image(x):
    plotdata = x / 255
    plt.gray()
    plt.imshow(plotdata)
    plt.show()

def sigmoid(x):
    return 1/(1+1/e**x)

def forward_propagation():

    
if __name__ == '__main__':
    data = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = data.load_data()
    x_train, x_test = x_train, x_test

    