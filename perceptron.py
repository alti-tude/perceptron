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


def initialise(dims):
    """
        dims contains the dimensions of the layers where the first layer is the input layer(direct inputs) 
        and the last is the output

        w[i] -> wt from (i-1)th layer to the ith layer
        b[i] -> b for the ith layer
    """
    l = len(dims)
    b = [[0]]
    w = [[0]]
    np.random.seed(5)
    for i in range(1,l):
        wt = np.random.randn(dims[i-1], dims[i])
        bt = np.zeros( (dims[i],1) )
        b.append(bt)
        w.append(wt)
    
    return w, b


def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward_propagation(l, x, w, b):
    """
        l -> number of layers(including input)
        x -> current test case (img_x * img_y * 1, 1)
        w -> wts (layer l, n[l-1], n[l])
    """
    a = [x]
    for i in range(1, l):
        z = w[i].T @ a[i-1] + b[i]
        a.append(sigmoid(z))
    return a


def cost(yhat, y):
    y = y.reshape(y.shape[0], 1)
    yhat = yhat.reshape(yhat.shape[0], 1)
    return -(y.T @ np.log(yhat) + (1-y).T @ np.log(1-yhat))[0][0]


def loss(l, x, y, w, b):
    """
        m -> number of training examples
        x -> the raw_training input (m, imgx, imgy)
        y -> is the training output (m, 1)
        w -> (l, dim[i-1], dim[i])
        b -> (l, dim[i], 1)
    """
    m = len(x)
    j = 0

    for i in range(m):
        batch = x[i]
        x_flat = batch.reshape(batch.shape[0]*batch.shape[1], 1)
        a = forward_propagation(l, x_flat, w, b)
        
        y_cur = np.zeros( (dims[-1], 1) )
        y_cur[y[i]] = 1
        j += cost(a[-1], y_cur)

        done = int(i/len(x)*100)
        print('#'*done + '_'*(100-done), end="\r")
   
    print("")
    print(j/m)
    return j/m


if __name__ == '__main__':
    data = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = data.load_data()
    x_train, x_test = x_train, x_test
    
    dims = [28*28, 10, 9, 10]
    w, b = initialise(dims)
    m = len(x_train)

    #cost function test
    # y = np.array([1, 0, 0, 1])
    # yhat = np.array([0.5, 0.1, 0.2, 0.001])
    # assert(abs(cost(yhat, y)[0][0] - 7.92940653) < 1e10)
    
    loss(4, x_train, y_train, w, b)
    