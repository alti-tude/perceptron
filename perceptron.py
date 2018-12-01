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
    return 1/(1+np.exp(-x))


def initialise(dims, x, y):
    """
        dims contains the dimensions of the layers where the first layer is the input layer(direct inputs) 
        and the last is the output

        w[i] -> wt from (i-1)th layer to the ith layer
        b[i] -> b for the ith layer

        x_flat -> (dims[0] x m)
        y_expanded -> (k x m)
    """
    l = len(dims)
    m = len(x)
    k = dims[-1]
    
    ############# w and b ###########################
    b = [[0]]
    w = [[0]]
    np.random.seed(5)
    for i in range(1,l):
        wt = np.random.randn(dims[i-1], dims[i])
        bt = np.zeros( (dims[i],1) )
        b.append(bt)
        w.append(wt)
    
    ############ x_flat ##############################
    x_flat = np.zeros( (m, dims[0]) ) #take transpose later for proper x_flat (dim[0]->ip layer  x  m)

    for i in range(m):
        batch = x[i].reshape(1, x[i].shape[0]*x[i].shape[1])
        x_flat[i] = batch
    
    x_flat = x_flat.T

    ######### y_cur ##################################
    y_expanded = np.zeros( (dims[-1], m) )
    
    for i in range(m):
        y_expanded[y[i]][i] = 1

    return w, b, x_flat, y_expanded ,l , m, k


def forward_propagation(l, x, w, b):
    """
        l -> number of layers(including input)
        x -> test case (img_x * img_y * 1, m)
        w -> wts (layer l, n[l-1], n[l])

        return a[-1] (k x m)
    """
    a = [x]
    for i in range(1, l):
        z = w[i].T @ a[i-1] + b[i]
        a.append(sigmoid(z))
    return a[-1]


def loss(yhat, y, k, m):
    """
        iterate over all the outputs and take effective dot product (@ takes internal dot)
        -(y[i] @ np.log(yhat[i]) + (1-y[i]) @ np.log(1-yhat[i])) -> cost
        
        yhat -> k x m
        y -> k x m

        return int
    """
    loss = 0
    for i in range(k):
        loss += -(y[i] @ np.log(yhat[i]) + (1-y[i]) @ np.log(1-yhat[i]))
    return loss/m


if __name__ == '__main__':
    data = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = data.load_data()
    x_train, x_test = x_train, x_test
    
    dims = [28*28, 10, 9, 10]
    w, b, x_flat, y_expanded, l, m, k = initialise(dims, x_train, y_train)
    print(x_flat.shape)
    print(x_train.shape)
    
    print(y_expanded.shape)
    yhat = forward_propagation(l, x_flat, w, b)
    print(loss(yhat, y_expanded, k, m))
    #cost function test
    # y = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    # y = y.T 
    # yhat = np.array([[0.5,0.5,0.5,0.5], [0.5,0.5,0.5,0.5]])
    # yhat = yhat.T
    # print(yhat)
    # print(cost(yhat, y))

    # assert(abs(cost(yhat, y)[0][0] - 7.92940653) < 1e10)
    
    # loss(4, x_train, y_train, w, b)
    