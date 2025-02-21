import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

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

def mean_wave(x_values, y_values, verbose=False):
    """
    :param x_values: numpy array of x values
    :param y_values: numpy array of y values
    :param verbose: boolean, controls the verbosity of the function
    :return: a tuple in the form (average wave y values, averaged wave x values)
    """
    wave_amplitude = y_values.max()-y_values.min()

    trough_indices, _ = find_peaks(-1*y_values, prominence=wave_amplitude/4)  # Negate y_values to find minima

    interpolated_waves = []
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label="Waveform", color='blue')
        plt.scatter(x_values[trough_indices], y_values[trough_indices], color='red', label='Troughs', zorder=5)
        plt.title("Waveform with Trough Points")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    for i in range(len(trough_indices) - 1):
        # Extract data for the current segment
        start_index = trough_indices[i]
        end_index = trough_indices[i + 1]
        x_segment = x_values[start_index:end_index]
        y_segment = y_values[start_index:end_index]

        # Shift x-coordinates for alignment (except the first wave)
        if i > 0:
            x_segment = x_segment - (x_segment[0] - x_values[trough_indices[0]])

        # Initialize the common x-axis using the first segment
        if i == 0:
            x_min = x_segment[0]
            x_max = x_segment[-1]
            x_common_points = len(x_segment)
            x_common = np.linspace(x_min, x_max, x_common_points)

        # Interpolate to the common x-axis
        interp_y = interp1d(x_segment, y_segment, kind='linear', fill_value="extrapolate")(x_common)
        interpolated_waves.append(interp_y)

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.title("Set of Waveforms")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        for wave_index, waveform in enumerate(interpolated_waves):
            plt.plot(x_common, waveform, label=f"Waveform {wave_index}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Convert the list of interpolated waves to a NumPy array for calculations
    interpolated_waves_np = np.vstack(interpolated_waves)

    # Calculate the initial average and standard deviation
    average_wave = np.mean(interpolated_waves_np, axis=0)
    std_wave = np.std(interpolated_waves_np, axis=0)

    # Filter out waves outside the range of average ± standard deviation
    threshold_percentage = 80

    filtered_waves = []
    for wave in interpolated_waves_np:
        # Calculate the percentage of points that meet the OR condition
        within_range = (wave >= (average_wave - std_wave)) & (
                    wave <= (average_wave + std_wave))  # Points above or equal to lower bound
        percentage_within_range = np.sum(within_range) / len(wave) * 100

        # Include the wave if the percentage is above the threshold
        if percentage_within_range >= threshold_percentage:
            filtered_waves.append(wave)

    # Recalculate the average and standard deviation with the filtered waves
    filtered_waves_np = np.vstack(filtered_waves)
    new_average_wave = np.mean(filtered_waves_np, axis=0)
    new_std_wave = np.std(filtered_waves_np, axis=0)
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.title("Average waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.plot(x_common, new_average_wave)
        plt.grid(True)
        plt.show()

    return new_average_wave, x_common