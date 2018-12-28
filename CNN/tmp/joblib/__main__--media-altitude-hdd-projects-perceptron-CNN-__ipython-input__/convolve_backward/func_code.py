# first line: 1
@mem.cache
def convolve_backward(dA, w, b, hparams, A_prev):
    m, nh, nw, nc = dA.shape
    m, nh_prev, nw_prev, nc_prev = A_prev.shape
    f, f, nc_prev, nc = w.shape
    stride = hparams["stride"]
    pad = hparams["pad"]
    
    dA_prev = np.zeros(A_prev.shape) 
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    for i in range(0, nh_prev-f+1, stride):
        for j in range(0, nw_prev-f+1, stride):
            for c in range(0,nc):
                dA_slice = dA[:,int(i/stride), int(j/stride), c]
                dA_slice = dA_slice.reshape(*dA_slice.shape, 1,1, 1)
                w_shifted = np.array([w[:,:,:,c]])
                
                dA_prev[:,i:i+f, j:j+f, :] += dA_slice*w_shifted
                dw[:,:,:,c] += np.sum(A_prev[:,i:i+f, j:j+f, :] * dA_slice, axis=0)
                db[:,:,:,c] += np.sum(dA_slice, axis=0)
                
    return dA_prev, dw, db
