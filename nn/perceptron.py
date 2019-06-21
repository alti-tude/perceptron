import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, pickle
e = np.e

class loss_plotter():
    def __init__(self):
        plt.ion()
        self.figure = plt.figure()
        self.losses = np.array([])
    
    def add_loss(self, loss):
        self.losses = np.append(self.losses, [loss])
        print(self.losses)
    
    def show(self):
        x = np.arange(max(0,self.losses.shape[0]-10) ,self.losses.shape[0], 1)
        y = self.losses[max(0,self.losses.shape[0]-10):]
        plt.clf()
        plt.plot(x,y)
        self.figure.canvas.draw()

def plot_image(x):
    plotdata = x / 255
    plt.gray()
    plt.imshow(plotdata)
    plt.show()


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a, a*(1-a)


def layer_function(func, z):
    if func == "SIGMOID":
        return sigmoid(z)


def initialise(dims, x, y, feat_norm):
    """
        dims contains the dimensions of the layers where the first layer is the input layer(direct inputs) 
        and the last is the output

        w[i] -> wt from (i-1)th layer to the ith layer
        b[i] -> b for the ith layer

        x_flat -> (dims[0] x m)
        y_expanded -> (k x m)

        delta -> empty l sized array of arrays
    """
    l = len(dims)
    m = len(x)
    k = dims[-1]
    
    ############# w, b, delta ###########################
    b = [[0]]
    w = [np.zeros( (1,1) )]
    delta = [[0]]
    np.random.seed(2)
    for i in range(1,l):
        wt = np.random.randn(dims[i-1], dims[i])
        bt = np.zeros( (dims[i],1) )
        b.append(bt)
        w.append(wt)
        delta.append([0])
    
    ############ x_flat ##############################
    x_flat = np.zeros( (m, dims[0]) ) #take transpose later for proper x_flat (dim[0]->ip layer  x  m)

    for i in range(m):
        batch = x[i].reshape(1, x[i].shape[0]*x[i].shape[1])
        x_flat[i] = batch / feat_norm

    x_flat = x_flat.T
    
    ######### y_cur ##################################
    y_expanded = np.zeros( (dims[-1], m) )
    
    for i in range(m):
        y_expanded[y[i]][i] = 1

    return w, b, delta, x_flat, y_expanded ,l , m, k


def forward_propagation(l, x, w, b, layer_funcs):
    """
        l -> number of layers(including input)
        x -> test case (img_x * img_y * 1, m)
        w -> wts (layer l, n[l-1], n[l])

        return a (l, dim[l] x m)
    """
    a = [x]
    g_prime = [x*(1-x)] #input layer is not used anyway so doesn't matter if you make it sigmoid independent
    for i in range(1, l):
        z = w[i].T @ a[i-1] + b[i]
        sigma = layer_function(layer_funcs[i], z)
        a.append(sigma[0])
        g_prime.append(sigma[1])
    
    return a, g_prime


def loss(yhat, y, k, m):
    """
        iterate over all the outputs and take effective dot product (@ takes hadamad product)
        -(y[i] @ np.log(yhat[i]) + (1-y[i]) @ np.log(1-yhat[i])) -> cost
        
        y[i,np.newaxis] returns the correct shape

        yhat -> k x m
        y -> k x m

        return int
    """
    loss = 0
    
    for i in range(k):
        loss += -(y[i, np.newaxis] @ np.log(yhat[i,np.newaxis].T) + (1-y[i,np.newaxis]) @ np.log(1-yhat[i,np.newaxis].T))
    return loss/m

def mse_loss(yhat, y, k, m):
    """
        yhat -> k x m (predicted)
        y -> k x m

        return int
    """
    loss = 1/(2*m)*np.sum(np.power(yhat - y,2))
    return loss


def back_propagation(l, a, g_prime, y, w, b):
    m = y.shape[1]
    # delta = 1/m*(a[-1] - y) * a[-1] * (1-a[-1])
    delta = (a[-1]-y)
    dw = [ np.zeros( (i.shape) ) for i in w ]
    
    for i in range(l-2, -1, -1):
        dw[i+1] = a[i] @ delta.T
        delta = w[i+1] @ delta * (g_prime[i])
    
    db = [ np.sum(i) for i in  delta ]
    return dw, db


def grad_decent(iter, alpha, m, l, k, w, b, x_flat, y_expanded, layer_funcs):
    x = x_flat
    loss_plt = loss_plotter()
    for i in range(iter):
        done = int(i/iter*100)

        a, g_prime = forward_propagation(l, x, w, b, layer_funcs)
        x = a[0]
        yhat = a[-1]
        dw, db = back_propagation(l, a, g_prime, y_expanded, w, b)
        
        os.system("clear")
        L = loss(yhat, y_expanded, k, m)
        print(L, end="   ")
        print("#"*done + "-"*(100-done) + str(i), end="\r")
        loss_plt.add_loss(L)
        loss_plt.show()

        for j in range(1,l):
            w[j] = w[j] - (alpha/m) * dw[j]
            b[j] = b[j] - (alpha/m) * db[j]
    
    print("\n")
    return w, b


def accuracy(k, l, x_flat, y, w, b, layer_funcs):
    yhat = forward_propagation(l, x_flat, w, b, layer_funcs)[0][-1]
    count = 0

    for i in range(m):
        ma = 0
        maidx = -1

        for j in range(k):
            if ma <= yhat[j][i]:
                ma = yhat[j][i]
                maidx = j

        if y[i] == maidx:
            count += 1

    print(count/m)
    return count/m


if __name__ == '__main__':
    if not os.path.exists('./mnist_dataset.pickle'):
        data = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = data.load_data()
        with open('./mnist_dataset.pickle', 'wb') as fil:
            pickle.dump(((x_train, y_train),(x_test, y_test)),fil, fix_imports=False)

    with open('./mnist_dataset.pickle', 'rb') as fil:
        ((x_train, y_train),(x_test, y_test)) = pickle.load(fil)

    dims = [28*28, 397, 100, 10]
    layer_funcs = ["SIGMOID", "SIGMOID", "SIGMOID", "SIGMOID"]

    w, b, delta, x_flat, y_expanded, l, m, k = initialise(dims, x_train, y_train, 255)

    # plot_image(x_test[101])

    w, b = grad_decent(500, 1, m, l, k, w, b, x_flat, y_expanded, layer_funcs)
    acc = accuracy(k,l,x_flat, y_train, w, b, layer_funcs)
    
    _, _, delta, x_flat, y_expanded, l, m, k = initialise(dims, x_test, y_test, 255)
    acc = accuracy(k,l,x_flat, y_test, w, b, layer_funcs)
