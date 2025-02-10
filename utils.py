import numpy as np
from pathlib import Path
def cubify(arr, roi_shape):
    """
    :param arr: array of values, intended to be 3D
    :param roi_shape: list or tuple of values defining the shape of the sub-array that you input array will be reshaped
    into
    :return: a num elems in array/num elems in arrays of shape.roi_shape by num elems in arrays of shape.roi_shape array
    where array[0][:] represents the elements in the 0th sub array of the input
    """
    oldshape = np.array(arr.shape)
    repeats = (oldshape / roi_shape).astype(int) # how many times each dimension of roi_shape repeat for each dimension
    # of the input array
    tmpshape = np.column_stack([repeats, roi_shape]).ravel() # a 1-D array, where the nth odd element are is the repeat
    # number for that dimension and the nth even element is the roi shape corresponding to that dimension
    order = np.arange(len(tmpshape)) # array used for reordering the permuted array in accordance with (repeats,
    # roi_shape)
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    arr = arr.reshape(tmpshape)
    arr = arr.transpose(order) # arr is snow of arr.shape = (repeats, roi_shape)
    arr = arr.reshape(-1, np.prod(roi_shape)) # reshaped to n rois, by n elems in roi
    return arr

def calculate_fractal_dimension(coeff_vars, window_sizes, type='Greyscale', dim=3):
    """
    :param coeff_vars:list of 1D array of floats, each coefficient of variation corresponds to the correesponding window
    size in the window_sizes array
    :param window_sizes:
    :param type: string determining the type of image that was analysed, either greyscale or binary
    :param dim:
    :return:
    """
    if type == 'Greyscale':
        grad, _ = np.polyfit(np.log(window_sizes), np.log(coeff_vars), 1)
        frac_dim = 1 - grad
    elif type == 'binary':

        logx = np.log(window_sizes)
        logy = np.log(coeff_vars)

        m, b = np.polyfit(logx, logy, 1)
        frac_dim = -1 * m
    return frac_dim

def cartesian_product(*arrays):
    """
    :param arrays:
    :return:
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def confirm_directory(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        print(f"{directory} did not exist, it has been created")
    return
