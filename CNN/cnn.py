import numpy as np 
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)

def plot_image(x):
    plotdata = x / 255
    plt.gray()
    plt.imshow(plotdata)
    plt.show()

def zero_pad(x, pad):
    np.pad(\
            x, ((0,0), (pad, pad), (pad, pad), (0,0)),\
            'constant', constant_values = (0)
        )