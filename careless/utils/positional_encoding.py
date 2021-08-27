import numpy as np

def positional_encoding(X, L):
    """
    The positional encoding as defined in the NeRF paper https://arxiv.org/pdf/2003.08934.pdf
      gamma(p) = (sin(2**0*pi*p), cos(2**0*pi*p), ..., sin(2**(L-1)*pi*p), cos(2**(L-1)*pi*p))
    Wherein p represents an arbitrary batched set of vectors computed by normalizing X between
    between -1 and 1
    """
    p = 2.*(X - X.min(-2)) / (X.max(-2) - X.min(-2)) - 1.
    L = np.arange(L, dtype=X.dtype)
    f = np.pi*2**L
    fp = (f[...,None,:]*p[...,:,None]).reshape(p.shape[:-1] + (-1,))
    return np.concatenate((
        np.cos(fp),
        np.sin(fp),
    ), axis=-1)


