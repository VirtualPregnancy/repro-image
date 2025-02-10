from reproimage.utils import cubify, calculate_fractal_dimension
import SimpleITK as sitk
from scipy.stats import variation
import numpy as np

def efficient_largest_ccmp_filter(image: sitk.Image):
    """
    :param image: with a single label
    :return: image with a single connected compenent with a value of 1 that corresponds to the single largest connected
    component in the input image
    """
    ccmp = sitk.ConnectedComponent(image) # initialise connected component filter
    stats = sitk.LabelShapeStatisticsImageFilter() # initialise label shape statistics filter
    stats.Execute(ccmp)
    label_sizes = []
    for label in stats.GetLabels(): #iterate over labels and create a list of label sizes
        label_sizes.append(stats.GetNumberOfPixels(label))
    label_max = label_sizes.index(max(label_sizes)) + 1 # value of max label is equal to the index of the max label size
    # plus one as the 0th label does not represent anything, and the 0th element of the label_sizes list represent the
    # size of the label valued 1
    image = ccmp == label_max # eleminate all labels except for the largest label
    return image

def Lacunarity(img, grid_spacings):
    """
    :param img: binary image
    :param grid_spacings: list of integers defining the spacing of fixed ND grids used to evaluate over the input image
    :return: a series of lacunarity values defined over the grid spacings
    """
    lacunarity = []
    for grid_spacing in grid_spacings: # iterate over grid spacings and evaluate the lacunarity at each value
        img_arr = sitk.GetArrayFromImage(img)

        split_axis = list(img_arr.shape)
        for n,n_split in enumerate(split_axis): # calculate how many times the grid will repeat for each dimension, with
            # the overflow for that division stored in a tuple
            split_axis[n] = (np.floor(n_split/grid_spacing), n_split%grid_spacing)

        img_arr = img_arr[split_axis[0][1]:, split_axis[1][1]:, split_axis[2][1]:] #trim the image from the overflow
        # values for each dimension to the size of that dimension such that the grid divides perfectly into the trimmed
        # image
        test_var = cubify(img_arr, [grid_spacing, grid_spacing, grid_spacing]) # array of grid values from the
        # image array

        test_var = np.sum(test_var, axis=1)/(grid_spacing**3) # array of means for each box in the grid

        lacunarity.append(variation(test_var)**2) # coefficient of variation for the mean through the fixed grid squared

        # print(f"The lacunarity for the length-scale of {grid_spacing}, is {variation(test_var)**2}")
    return lacunarity

def surface_area(input_im):
    """
    :param input_im: binary sitk.Image with isotropic spacing with the label having a value of 1
    :return: surface area of the binary label not including faces on the edge of the image space, surface area is
    calculated on a voxel basis and then scaled to the image spacing.
    """
    native_size = input_im.GetSize()
    mirror_image_boundary = sitk.MirrorPadImageFilter()
    mirror_image_boundary.SetPadLowerBound([1, 1, 1])
    mirror_image_boundary.SetPadUpperBound([1, 1, 1])
    padded_im = mirror_image_boundary.Execute(input_im)
    kernel = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    kernel[1, 1, 0] = 1
    kernel[2, 1, 1] = 1
    kernel[0, 1, 1] = 1
    kernel[1, 2, 1] = 1
    kernel[1, 1, 2] = 1
    kernel[1, 0, 1] = 1

    conv_im = sitk.Convolution(padded_im, kernel)
    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(native_size)
    extract_filter.SetIndex([1, 1, 1])

    out_im = extract_filter.Execute(conv_im)
    out_im.CopyInformation(input_im)
    out_im = out_im * (input_im == 0)
    area = sitk.GetArrayFromImage(out_im).flatten().sum()
    area = area * (input_im.GetSpacing()[0] * input_im.GetSpacing()[0])

    return area

def binary_fractal_window_series(img, window_sizes):
    """
    :param img: binary sitk.Image
    :param window_sizes: list of integers
    :return: fractal dimension for the input image calculated over the window sizes provided using a fixed grid
    algorithm
    """
    fractal_dim_series = []
    n_grid_samples = []
    for grid_spacing in window_sizes:
        img_arr = sitk.GetArrayFromImage(img)
        split_axis = list(img_arr.shape)
        for n,n_split in enumerate(split_axis):
            split_axis[n] =  (np.floor(n_split/grid_spacing), n_split%grid_spacing)

        img_arr = img_arr[split_axis[0][1]:, split_axis[1][1]:, split_axis[2][1]:]
        test_var = cubify(img_arr, [grid_spacing, grid_spacing, grid_spacing])

        sample_window_range = np.mean(test_var, axis=1)
        count = np.count_nonzero(sample_window_range)

        fractal_dim_series.append(count)

    frac_dim = calculate_fractal_dimension(fractal_dim_series, window_sizes, type='binary')
    return frac_dim

def n_largest_components_filter(img, n_components = 0):
    ccmp = sitk.ConnectedComponent(img)
    ccmp_sort = sitk.RelabelComponent(ccmp, sortByObjectSize = True)
    largest_comp = sitk.Threshold(ccmp_sort, upper=float(n_components), outsideValue=float(0))
    largest_comp = sitk.Cast(largest_comp, sitk.sitkUInt8)

    return largest_comp