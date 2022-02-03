import numpy as np
import scipy.signal as sp
from scipy import ndimage


def convolve(xarr, weights, dim):
    axis = xarr.get_axis_num(dim)
    depth = [0] * xarr.ndim
    depth[axis] = len(weights)//2 + 1

    def func(x):
        return ndimage.convolve1d(x, weights, axis=axis, mode="constant")
    data = xarr.data.map_overlap(func, depth=depth, boundary=0, dtype="float")
    return xarr.copy(data=data)


def filter(xarr, dim, ftype, cutoff, numtaps, width=None, window='hamming', minimum_phase=False):
    delta = xarr[dim][1] - xarr[dim][0]
    if np.issubdtype(delta, np.timedelta64):
        delta = delta / np.timedelta64(1, "s")
    fs = 1 / delta
    weights = sp.firwin(numtaps, cutoff, width=width,
                        window=window, pass_zero=ftype, fs=fs)
    if minimum_phase:
        weights = sp.minimum_phase(weights)
    return convolve(xarr, weights, dim)
