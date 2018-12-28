# first line: 1
@mem.cache
def pool_backward(dA, hparams, A_prev):
    m, nh, nw, nc = dA.shape
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    stride = hparams["stride"]
    f = hparams["f"]
    dA_prev = np.zeros(A_prev.shape) 
     
    for i in range(0, nh_prev-f+1, stride):
        for j in range(0, nw_prev-f+1, stride):
            for c in range(0,nc):
                dA_slice = dA[:,int(i/stride), int(j/stride), c]
                dA_slice = dA_slice.reshape(*dA_slice.shape, 1,1, 1)
                mask = create_mask(A_prev[:,i:i+f, j:j+f, :])
                
                dA_prev[:,i:i+f, j:j+f, :] += dA_slice*mask
                
    return dA_prev
