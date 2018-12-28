# first line: 1
@mem.cache
def zero_pad(x, pad):
    return np.pad(
            x, 
            ((0,0), (pad, pad), (pad, pad), (0,0)),
            'constant', 
            constant_values = (0)
        )
