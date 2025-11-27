from reproimage.utils import cubify, calculate_fractal_dimension, get_image_metadata_from_log_file, Suppressor
import SimpleITK as sitk
from scipy.stats import variation
import numpy as np
import re, os, sys
from math import ceil

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

def image_stack_to_volume(reconstruction_directory, output_directory, sample_identifier, isotropic_downsample_dim=8,
                    grid_size=[8, 8, 8], specified_mosaic_piece_size=[], raw_im_stack_size=[], raw_im_stack_spacing=[]):
    if not os.path.isdir(output_directory):
        print(f"Error code 2: Output directory provided is not a real directory ({output_directory})")
        exit(2)

    # script parameters
    grid_x, grid_y, grid_z = grid_size

    root_dir = reconstruction_directory
    f_name = output_directory
    f_identifier = sample_identifier

    grid_map = sitk.Image([grid_z, grid_x, grid_y], sitk.sitkUInt16)

    if root_dir == '':
        print("Script terminated as no directory was provided.")
        exit(2)

    # calculate image parameters for the output image
    try:
        log_file = os.path.join(root_dir,
                                [f for f in os.listdir(root_dir) if f.endswith('rec.log') and not f.startswith('._')][
                                    0])
        raw_im_stack_size, raw_im_stack_spacing, fileformat = get_image_metadata_from_log_file(log_file)
        n_images = raw_im_stack_size[2]
    except IndexError:
        print(
            "No log file available, script will run with no spatial information, and with the best possible approximation of stack size using the files contained within your chosen directory")
        n_images = len(os.listdir(root_dir))
        if raw_im_stack_spacing == []:
            raw_im_stack_spacing = [1, 1, 1]
        # image size parameters based off first image in the stack
        im_address_for_params = os.path.join(root_dir, os.listdir(root_dir)[-1])
        with Suppressor():
            im_for_params = sitk.ReadImage(im_address_for_params)
        if raw_im_stack_size == []:
            raw_im_stack_size = list(im_for_params.GetSize())
            raw_im_stack_size.append(n_images)
        fileformat = '.' + os.path.split(im_address_for_params)[-1].split('.')[-1]

    # create list of filenames within target directory
    filelist = [f for f in os.listdir(root_dir) if f.endswith(fileformat)]  # tif code
    filelist.sort(key=lambda f: int(re.sub('\D', '', f)), reverse=True)
    filelist = list(filter(lambda f: not (f.endswith('spr' + fileformat)), filelist))  # tif code
    filelist = list(filter(lambda f: not ('pp' in f), filelist))  # tif code
    filelist = list(filter(lambda f: not ('prev' in f), filelist))  # tif code

    if len(filelist) == 0:
        print("There are no suitable files in this directory")
        exit(1)

    resamplinglayer_size = isotropic_downsample_dim
    number_of_stacks_in_mosaic_layer = ceil(n_images / resamplinglayer_size)
    stack_remainder = n_images % resamplinglayer_size

    output_spacing = tuple([dim_spacing * isotropic_downsample_dim for dim_spacing in raw_im_stack_spacing]) * 3
    mosaic_piece_size = [0.0, 0.0, 0.0]
    output_size_ds = [0, 0, 0]

    output_size_ds[0] = int(raw_im_stack_size[0] * raw_im_stack_spacing[0] / output_spacing[0] + .5)
    output_size_ds[1] = int(raw_im_stack_size[1] * raw_im_stack_spacing[1] / output_spacing[1] + .5)
    output_size_ds[2] = int(raw_im_stack_size[2] * raw_im_stack_spacing[2] / output_spacing[2] + .5)

    if specified_mosaic_piece_size != []:
        output_size_ds[0] = output_size_ds[0] + (
                    specified_mosaic_piece_size[0] - output_size_ds[0] % specified_mosaic_piece_size[0])
        output_size_ds[1] = output_size_ds[1] + (
                    specified_mosaic_piece_size[1] - output_size_ds[1] % specified_mosaic_piece_size[1])
        output_size_ds[2] = output_size_ds[2] + (
                    specified_mosaic_piece_size[2] - output_size_ds[2] % specified_mosaic_piece_size[2])

    mosaic_piece_size[0] = int(output_size_ds[0] / grid_x)
    mosaic_piece_size[1] = int(output_size_ds[1] / grid_y)
    mosaic_piece_size[2] = int(output_size_ds[2] / grid_z)
    im_size = os.stat(os.path.join(root_dir, filelist[1])).st_size  # size of one full size .bmp image in bytes

    # resampling summary info
    print(f'There are {n_images} \'.{fileformat}\' files in the reconstruction directory')
    print(f'The entire image stack is {round(im_size * n_images / 1024 ** 3, 2)} GB')
    print(
        f'The stack is composed of {n_images} images that are {raw_im_stack_size[0]} by {raw_im_stack_size[1]} pixels, with spacing of {raw_im_stack_spacing} µm')

    # create output image
    result_img = sitk.Image(mosaic_piece_size, 1, 1)  # SIZE, pixelID, numberOfComponentsPerPixel
    result_img.SetSpacing(output_spacing)
    result_img.SetOrigin([0.0, 0.0, 0.0])
    result_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # identity matrix

    ds_size = [output_size_ds[0], output_size_ds[1], mosaic_piece_size[2]]

    # initialise resample image filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # Identity matrix
    resample.SetOutputSpacing(output_spacing)
    resample.SetOutputOrigin([0.0, 0.0, 0.0])
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    extract = sitk.ExtractImageFilter()
    Reader = sitk.ImageSeriesReader()
    #######TEMP
    debug = False
    origins = []
    lin_idx = 0
    global_z_layer = 0  # work variable to keep track of what z slice the resampling algorithm is up to, specifically for when the resampled image stacks are added into the output image dataframe

    # resample the original images stack by stack
    print("Resampling image data")
    print(
        f"The raw reconstruction image stack will be sliced into stacks of {resamplinglayer_size} images for the resampling process")
    print("Processing . . . ")

    for z_grid_index in range(0, grid_z):

        downsample_img = sitk.Image(ds_size, 1, 1)  # SIZE, pixelID, numberOfComponentsPerPixel
        downsample_img.SetSpacing(output_spacing)
        downsample_img.SetOrigin([0.0, 0.0, 0.0])
        downsample_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # identity matrix

        images_left_in_stack = len(filelist)
        if images_left_in_stack > mosaic_piece_size[2] * isotropic_downsample_dim:
            images_remaining = mosaic_piece_size[2] * isotropic_downsample_dim  # -skip_end# n_images
        else:
            images_remaining = images_left_in_stack
        number_of_stacks_in_mosaic_layer = mosaic_piece_size[2] * isotropic_downsample_dim / resamplinglayer_size
        stack_number = 1

        while images_remaining > 0:  # this while loop loads in the entirety of the mosaic layer and either resamples it
            # or does not, depending on the resmapling factor chosesn in the function execution
            if images_remaining < resamplinglayer_size:
                resamplinglayer_size = images_remaining
            images_remaining -= resamplinglayer_size
            if debug:
                print('Images remaining', images_remaining)
            # pop names off the sorted list of image filenames, used to parameterize the reader object
            names = []  # clear the names list
            for n in range(0, resamplinglayer_size):  # each images i na stack
                names.append(os.path.join(root_dir, filelist.pop(-1)).replace("\\", "/"))
            if debug:
                print('Names', names)
            Reader.SetFileNames(names)
            with Suppressor():
                temp_image = Reader.Execute()  # read in stack of images

            if temp_image.GetPixelIDValue() != sitk.sitkUInt8:
                if temp_image.GetPixelIDValue() == 13:
                    temp_image = sitk.VectorIndexSelectionCast(temp_image, 0, 1)

                if temp_image.GetPixelIDValue() != sitk.sitkUInt8:
                    temp_image = sitk.RescaleIntensity(temp_image, 0, 255)
                    temp_image = sitk.Cast(temp_image, sitk.sitkUInt8)

            temp_image.SetSpacing(
                raw_im_stack_spacing)  # bitmaps have no inherent spatial information, so it needs to be set using data read from the log file\

            if isotropic_downsample_dim > 1:  # this section encapsulates the bulk of what is needed to do the image resample if not mosaicing
                res_stack_size = ds_size[:2]
                res_stack_size.append(int(temp_image.GetSize()[2] * raw_im_stack_spacing[2] / output_spacing[2] + .5))
                resample.SetSize(res_stack_size)
                temp_downsize = resample.Execute(temp_image)

                # write downsampled stack into output image
                pasteFilter = sitk.PasteImageFilter()
                pasteFilter.SetSourceSize(temp_downsize.GetSize())
                pasteFilter.SetSourceIndex([0, 0, 0])
                pasteFilter.SetDestinationIndex(
                    [0, 0, stack_number - 1])  # use of the global z layer should only happen when grid size is 1

                # write resampled sub-stack into the output image
                downsample_img = pasteFilter.Execute(downsample_img, temp_downsize)
            else:
                # just write downsampled stack into output image
                pasteFilter = sitk.PasteImageFilter()
                pasteFilter.SetSourceSize(temp_image.GetSize())
                pasteFilter.SetSourceIndex([0, 0, 0])
                pasteFilter.SetDestinationIndex([0, 0, stack_number - 1])
                downsample_img = pasteFilter.Execute(downsample_img, temp_image)

            global_z_layer += int(resamplinglayer_size / isotropic_downsample_dim)  # update current z-slice level

            print(
                f"\rProcessed Image stack {stack_number - 1}/{number_of_stacks_in_mosaic_layer}, in layer: {z_grid_index + 1}/{grid_z}, {lin_idx + 1}/{grid_x * grid_y * grid_z} pieces have been written to file.",
                end='')
            sys.stdout.flush()

            stack_number += 1  #

        start_x = 0
        for i in range(0, grid_x):
            start_y = 0
            for j in range(0, grid_y):
                # lin_idx programmed to incremented by grid_size**2 as we move down each mosaic level down the z-index
                lin_idx = z_grid_index * (grid_x * grid_y) + i * (grid_y) ** 1 + j * (grid_y) ** 0
                grid_map[z_grid_index, i, j] = lin_idx
                # Select same subregion using ExtractImageFilter
                extract.SetSize([mosaic_piece_size[0], mosaic_piece_size[1], mosaic_piece_size[2]])
                extract.SetIndex([start_x, start_y, 0])
                # print([start_x, start_y, global_layer])
                origins.append((lin_idx, [start_x * output_spacing[0], start_y * output_spacing[1],
                                          z_grid_index * number_of_stacks_in_mosaic_layer * output_spacing[2]]))

                # write resampled sub-stack into the output image
                result_img = extract.Execute(downsample_img)

                # write resampled image to file, TODO: improve readability of origin handling
                result_img.SetMetaData("xyzt_units", "3")
                result_img.SetMetaData("origin", str([start_x, start_y,
                                                      z_grid_index * number_of_stacks_in_mosaic_layer * isotropic_downsample_dim]))
                result_img.SetOrigin([start_x * output_spacing[0], start_y * output_spacing[1],
                                      z_grid_index * number_of_stacks_in_mosaic_layer * output_spacing[2]])
                result_img.SetSpacing(output_spacing)
                # statistics_image_filter.Execute(result_img)
                # print(statistics_image_filter.GetMaximum())
                write = True
                if write:
                    try:
                        append = f_identifier
                        im_f_name = os.path.join(f_name, str(lin_idx) + '_' + append + '.nii')
                        sitk.WriteImage(result_img, im_f_name)
                        print(
                            f"\rProcessed Image stack {stack_number - 1}/{number_of_stacks_in_mosaic_layer}, in layer: {z_grid_index + 1}/{grid_z}, {lin_idx + 1}/{grid_x * grid_y * grid_z} pieces have been written to file.",
                            end='')
                        sys.stdout.flush()
                    except RuntimeError:
                        append = str(lin_idx) + '_'
                        print("Filename provided is invalid")
                        print(f"Saving as {lin_idx}_resampled_stack.nii")
                        im_f_name = append + 'resampled_stack.nii'
                        sitk.WriteImage(result_img, im_f_name)

                start_y = start_y + mosaic_piece_size[1]
            start_x = start_x + mosaic_piece_size[0]
    if grid_size != [1, 1, 1]:
        with open(os.path.join(output_directory, f_identifier + '_mosaic_mapping.txt'), 'w') as f:
            for mosaic_piece in origins:
                f.write(f'{mosaic_piece[0]}, {mosaic_piece[1]}\n')
            f.write(f'mosaic piece spacing, {output_spacing}\n')
            f.write(f'total mosaic size, {output_size_ds}\n')
            f.write(f'mosiac piece size, {mosaic_piece_size}\n')
            f.write(f'Grid size, {grid_size}\n')
            f.write(f'identifier,{f_identifier}\n')
            f.write(f'original reconstruction location, {root_dir}\n')
            f.write(f'downsampling factor, {isotropic_downsample_dim}\n')
    else:
        with open(os.path.join(output_directory, f_identifier + '_resampling_summary.txt'), 'w') as f:
            f.write(f'resampled image spacing, {output_spacing}\n')
            f.write(f'resampled image size, {output_size_ds}\n')
            f.write(f'Grid size, {grid_size}\n')
            f.write(f'identifier,{f_identifier}\n')
            f.write(f'original reconstruction location, {root_dir}\n')
            f.write(f'downsampling factor, {isotropic_downsample_dim}\n')


def downsample(reconstruction_directory, output_directory, sample_identifier, isotropic_downsample_dim):
    return image_stack_to_volume(reconstruction_directory, output_directory, sample_identifier, isotropic_downsample_dim,
                           grid_size=[1, 1, 1])