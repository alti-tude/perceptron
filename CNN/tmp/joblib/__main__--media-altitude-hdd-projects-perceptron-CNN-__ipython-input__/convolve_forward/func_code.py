# first line: 1
@mem.cache
def convolve_forward(A_prev, w, b, hparams):
    
    stride = hparams["stride"]
    pad = hparams["pad"]
    
    A_prev_pad = zero_pad(A_prev, pad)
    m, nh_prev, nw_prev, nc_prev = A_prev_pad.shape
    f, f, nc_prev, nc = w.shape
    
    nh = int((nh_prev - f)/stride) + 1
    nw = int((nw_prev - f)/stride) + 1
    Z = np.zeros((m, nh, nw, nc))
        
    for i in range(0,nh_prev-f+1,stride):
        for j in range(0,nw_prev-f+1,stride):
            A_slice = A_prev_pad[:, i:i+f, j:j+f, :]
            A_slice = A_slice.reshape(*A_slice.shape,1)
            w_shifted = np.array([w[:,:,:,:]])

            x = np.sum( A_slice * w_shifted, axis=(1,2,3))
            Z[:, int(i/stride), int(j/stride), :] = x + b[:,:,:,:]
    
    return Z
