# first line: 1
@mem.cache
def max_pool_forward(A_prev, hparams):
    stride = hparams["stride"]
    f = hparams["f"]
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    
    nh = int((nh_prev - f)/stride) + 1
    nw = int((nw_prev - f)/stride) + 1
    nc = nc_prev
    
    Z = np.zeros((m, nh, nw, nc))
    
    for i in range(0,nh_prev-f+1,stride):
        for j in range(0,nw_prev-f+1,stride):
            for c in range(0,nc):
                A_slice = A_prev[:, i:i+f, j:j+f, c]                
                Z[:, int(i/stride), int(j/stride), c] = np.max(A_slice)
    
    return Z
