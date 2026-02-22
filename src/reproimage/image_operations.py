from reproimage.utils import cubify, calculate_fractal_dimension, get_image_metadata_from_log_file, Suppressor
import SimpleITK as sitk
from scipy.stats import variation
import numpy as np
import re, os, sys
from math import ceil
from functools import partial
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

    output_spacing = tuple([dim_spacing * isotropic_downsample_dim for dim_spacing in raw_im_stack_spacing])

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

def get_subregion(img: sitk.Image, origin, size):
    """ This takes in a roi using voxel coordinates, not imagespace, it needs some error catching for regions of interest
    that are beyond the bounds of an image"""

    origin = np.array([x for x in origin])
    for count, index in enumerate(origin):
        if index<0:
            origin[count] = 0

    pt = origin.astype(np.uint16)
    if type(size) == int:
        reg = pt.tolist() + [int(size)]*3
    else:
        reg = pt.tolist() + [int(x) for x in size]

    count = 0
    for start,span, limit in zip(reg[:3],reg[3:], img.GetSize()):
        if start+span > limit:
            new_span = limit-start - 1
            reg[3+count] = int(new_span)
        count += 1
    reg = tuple(reg)
    ex_filter = sitk.ExtractImageFilter()
    ex_filter.SetIndex(reg[:3])
    ex_filter.SetSize(reg[3:])
    return ex_filter.Execute(img)

"""This code outlines the mosaic class for working with discretised large medical images"""
from toby_utils import get_subregion
class mosaic:
    def __init__(self, directory):
        # TO DO check directory validity
        self.directory = directory
        # search dir for mosaic_mapping file
        mapfile = ""
        mapping_file = 'mosaic_mapping.txt'
        file_list = os.listdir(directory)
        while file_list:
            mapfile = file_list.pop()
            if mapping_file in mapfile and mapfile[0] != '.':
                mapping_file = mapfile
                self.mapping_file = mapfile
                break
        if mapfile != mapping_file:
            print(f"cannot initialise mosaic object without spatial mapping information provided by {mapping_file}")
            raise SystemExit
        # make some kind of error code pop up

        self.elements = {}
        self.process_mapping_file(mapfile)  # this function reads the spatial mapping information in file,
        # setting the elements, and spacing attributes of the mosaic class
        self.n_pieces = max(self.elements.keys())+1 # adding one for zero indexing
        self.pieces = {}
        self.create_adjacency_matrix()
        self.function_history = []

    # noinspection PyAttributeOutsideInit
    def process_mapping_file(self, file):
        """
        Parameters
        ----------
        @file - mapping_file.txt

        Returns
        -------

        """
        file_path = os.path.join(self.directory, file)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                try:
                    a, b = line.split(sep=',', maxsplit=1)
                    if a.isnumeric():
                        b = b.replace('[', "")
                        b = b.replace(']', "")
                        b = b.replace('(', "")
                        b = b.replace(')', "")
                        b = b.split(',')
                        self.elements[int(a)] = tuple([float(x) for x in b])
                    elif 'spacing' in a:
                        b = b.replace('(', "")
                        b = b.replace(')', "")
                        b = b.split(',')
                        self.spacing = tuple([float(x) for x in b])
                    elif a == 'total mosaic size':
                        b = b.replace('[', "")
                        b = b.replace(']', "")
                        b = b.split(',')
                        self.macro_size = [int(x) for x in b]
                    elif 'original' in a:
                        self.original_directory = b
                    elif 'factor' in a:
                        self.downsample_factor = int(b)
                    elif a == 'mosiac piece size':
                        b = b.replace('[', "")
                        b = b.replace(']', "")
                        b = b.split(',')
                        self.piece_size = [int(x) for x in b]
                    elif a == 'Grid size':
                        b = b.replace('[', '')
                        b = b.replace(']', '')
                        b = b.split(',')
                        if len(b) == 1:
                            # for historic mosaic objects with isotropic/single parameter grid sizing
                            self.grid_size = tuple([int(b[0]) for x in range(0,3)])
                        else:
                            self.grid_size = tuple([int(x) for x in b])
                    elif a == 'identifier':
                        self.identifier = b.strip()
                except:
                    print(line)

            write_flag = False
            for prop in ['spacing', 'macro_size', 'original_directory', 'downsample_factor', 'piece_size', 'grid_size',
                         'identifier']:
                if not hasattr(self, prop):
                    self.prompt_for_attribute(prop)
                    write_flag = True
            if write_flag:
                self.write_mosaic_mapping_file(self.directory)

    # @property
    # def grid_size(self):
    #     print("Getting value...")
    #     return self.grid_size
    #
    # @grid_size.setter
    # def grid_size(self, value):
    #     print("Setting value...")
    #     self.grid_size = int(value)

    def prompt_for_attribute(self, prop):
        prop_dict = {'spacing': 'spacing', 'macro_size': 'total mosaic size',
                     'original_directory': 'original reconstruction location',
                     'down sample_factor': 'down sampling factor', 'piece_size': 'mosaic piece size',
                     'grid_size': 'Grid size', "identifier": "identifier"}
        if prop != 'identifier':
            print(
                f'Mosaic {self.identifier} does not contain information pertaining to {prop_dict[prop]}, would you '
                f'like to manually enter this information (y/n): ')
        else:
            print(
                f'Mosaic does not contain information pertaining to {prop_dict[prop]}, would you '
                f'like to manually enter this information (y/n): ')

        x = input()
        if x not in ['y', 'n', 'Y', 'N']:
            print(
                f'You have not provided acceptable input, please re-enter whether you would like'
                f' to manually specify the {prop_dict[prop]} (y/n):')
        if x in ['y', 'Y']:
            print('Please specify the property, don\'t make any mistakes!')
            x = input()
            if prop == 'grid_size':
                x = int(x)

            setattr(self, prop, x)
        else:
            return

    def read_mosaic_piece(self, indx):  # should probably think of a file naming convetion to match that outp
        """

        Parameters
        ----------
        indx

        Returns
        -------

        """
        file_name = ""
        file_list = os.listdir(self.directory)
        for x in file_list:
            if x.split(sep='_', maxsplit=1)[0] == str(indx):
                file_name = x

        if file_name == "":
            # split works from left to right
            # print(f"unable to read in mosaic piece {indx}")
            return 1

        file_path = os.path.join(self.directory, file_name)

        self.pieces[indx] = sitk.ReadImage(file_path)
        if self.pieces[indx].GetSpacing() != self.spacing:
            # print("re-wrote spacing due to mismatch")
            self.pieces[indx].SetSpacing(self.spacing)
            # print("re-wrote origin due to mismatch")
            self.pieces[indx].SetOrigin(self.elements[indx])
        return 0

    def load_all_pieces(self):
        for indx in self.elements.keys():
            self.read_mosaic_piece(indx)

    def unload_mosaic_piece(self, indx):
        self.pieces.pop(indx)

    def macro_histogram(self, nbins=10):
        for indx in self.elements.keys():
            if indx not in self.pieces.keys():
                self.read_mosaic_piece(indx)
            img_arr = sitk.GetArrayFromImage(self.pieces[indx])
            if 'bins' not in locals():
                counts, bins = np.histogram(img_arr.flatten(), bins=nbins, range = (0, 255))
            else:
                counts += np.histogram(img_arr.flatten(), bins=nbins, range = (0, 255))[0]
            self.unload_mosaic_piece(indx)
            print(f"\r{self.identifier}, piece {indx} processed in the mosaic wide histogram", ending='')
        print('\n')
        return counts, bins

    def apply_function_over_mosaic(self, fcn_hndle, append_string='', output_dir = ""):

        if output_dir == "":
            output_dir = self.directory

        if append_string == '':
            append_string = str(fcn_hndle)

        for indx in self.elements.keys():
            if indx not in self.pieces.keys():
                self.read_mosaic_piece(indx)

            img_f = fcn_hndle(self.pieces[indx])
            output_path = os.path.join(output_dir, self.identifier  + '-' + append_string)
            if os.path.isdir(output_path):
                output_file_path = os.path.join(output_path, str(indx) + '_' + append_string + '.nii')
            else:
                os.mkdir(output_path)
                output_file_path = os.path.join(output_path, str(indx) + '_' + append_string + '.nii')
            sitk.WriteImage(img_f, output_file_path)
            self.unload_mosaic_piece(indx)
            print(f'\r{self.identifier}, mosaic piece {indx} processed',end='')

        self.write_mosaic_mapping_file(output_path, append_text=append_string)
        print(f"\n{self.identifier} Mosaic processed succesfully")
        return output_path

    def apply_function_over_pieces(self, fcn_hndle, indices, append_string='', external_output=True):
        if not external_output:
            output = []
        for indx in indices:
            if indx not in self.pieces.keys():
                self.read_mosaic_piece(indx)

            img_f = fcn_hndle(self.pieces[indx])
            # print(sitk.GetArrayViewFromImage(img_f).max())
            if external_output:
                output_path = os.path.join(self.directory, self.identifier + '_' + append_string)
                if os.path.isdir(output_path):
                    output_file_path = os.path.join(output_path, str(indx) + '_' + append_string + '.nii')
                else:
                    os.mkdir(output_path)
                    output_file_path = os.path.join(output_path, str(indx) + '_' + append_string + '.nii')
                sitk.WriteImage(img_f, output_file_path)
            else:
                output.append(img_f)
            self.unload_mosaic_piece(indx)
            # print(f'Mosaic piece {indx} processed using the "{fcn_hndle}" function')
        if not external_output:
            return output

    def write_mosaic_mapping_file(self, directory, append_text=''):

        output_file = os.path.join(directory, self.identifier + append_text + '_mosaic_mapping.txt')
        with open(output_file, 'w') as f:
            for indx in self.elements.keys():
                f.write(f'{indx}, {self.elements[indx]}\n')
            f.write(f'mosaic piece spacing, {self.spacing}\n')
            f.write(f'total mosaic size, {self.macro_size}\n')
            f.write(f'mosiac piece size, {self.piece_size}\n')
            f.write(f'Grid size, {list(self.grid_size)}\n')
            f.write(f'identifier, {self.identifier}\n')
            f.write(f'original reconstruction location, {self.original_directory}\n')
            f.write(f'downsampling factor, {self.downsample_factor}\n')

    def visualise_piece(self, indx):
        sitk.Show(self.pieces[indx])

    # def mapping(self):
    #     grid_map = sitk.Image([grid_size, grid_size, grid_size], sitk.sitkUInt16)

    def create_adjacency_matrix(self):
        """
        Trying to remember the indexing from the creation of the initial mosaic
        Returns
        -------

        """
        self.adjacency_matrix = sitk.Image([self.grid_size[0], self.grid_size[1], self.grid_size[2]], sitk.sitkUInt16)
        adjacency_spacing = [x * y for x, y in zip(self.spacing, self.piece_size)]
        self.adjacency_matrix.SetSpacing(adjacency_spacing)
        self.adjacency_matrix.SetOrigin([x/2.0 -1 for x in adjacency_spacing])

        for i in range(0, self.grid_size[0]):
            for j in range(0, self.grid_size[1]):
                for k in range(0, self.grid_size[2]):
                    lin_idx = i * self.grid_size[1] ** 1 + j * self.grid_size[1] ** 0 + k * (
                            self.grid_size[0] * self.grid_size[1]) ** 1
                    self.adjacency_matrix[i, j, k] = lin_idx

    def location_of_point(self, point):
        """
        This function returns the mosaic priece that corresponds to a particular physical point in the total mosaic
        """
        indx = self.adjacency_matrix.TransformPhysicalPointToIndex(point)
        return self.adjacency_matrix.GetPixel(indx)

    def adjacent_pieces(self, piece_indx):
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetForegroundValue(piece_indx)
        dilate_filter.SetKernelType(sitk.sitkBox)
        # dilate_filter.SetKernelRadius(self.adjacency_matrix.GetSpacing())
        dilate_filter.SetKernelRadius(1)
        neighborhood = dilate_filter.Execute(self.adjacency_matrix)

        neighborhood = (neighborhood != piece_indx) * (piece_indx + 1)
        neighborhood = sitk.Mask(self.adjacency_matrix, neighborhood, maskingValue=piece_indx + 1,
                                 outsideValue=self.grid_size[0] * self.grid_size[1] * self.grid_size[2])
        pieces = []
        for item in sitk.GetArrayFromImage(neighborhood).flatten():
            if item != self.grid_size[0] * self.grid_size[1] * self.grid_size[2]:
                pieces.append(item)
        return pieces

    def get_piece_indices_for_sub_region(self, region):
        """

        Parameters
        ----------
        region [x,y,z,x_length, y_length, z_length]

        Returns
        -------

        """
        # convert region into pixels -
        x,y,z = self.adjacency_matrix.TransformPhysicalPointToIndex(region[:3])
        x_size, y_size, z_size = np.ceil([x/z for x,z in zip(region[3:], self.adjacency_matrix.GetSpacing())]).astype(int)
        pixel_region = [x,y,z, x_size, y_size, z_size]
        centroid = self.adjacency_matrix.TransformIndexToPhysicalPoint((x,y,z))
        for count, tup in enumerate(zip(region[:3], self.adjacency_matrix.GetSpacing(), centroid)):
            loc = tup[0]
            spacing = tup[1]
            cen_loc = tup[2]
            if loc>cen_loc:
                pixel_region[count+3] += 1
        # print(f"pixel_region: {pixel_region}")
        sub_region = get_subregion(self.adjacency_matrix, pixel_region[:3], pixel_region[3:])
        return sitk.GetArrayFromImage(sub_region).flatten()
    def get_sub_region(self, region):
        """
        :param region: region defined in terms of spatial origin, and distance from origin
        :return: a sub image from the greater mosaic structure with the origin and spatial size defined by the input region
        """
        sub_indices = self.get_piece_indices_for_sub_region(region)
        print(len(sub_indices))
        broad_region = self.create_mulitpiece_image(sub_indices)
        # convert region to pixels??
        origin = broad_region.TransformPhysicalPointToIndex(region[:3])
        span = [x/y for x,y in zip(region[3:], broad_region.GetSpacing())]
        span = np.ceil(span).astype(int)
        print(span, origin)
        sub_img = get_subregion(broad_region, origin, span)
        return sub_img


    def create_mulitpiece_image(self, indices):
        n_indices = len(indices)
        origs = np.zeros((n_indices, 3))
        for count, index in enumerate(indices):
            origs[count,:] = self.elements[index]
        min_orig = origs.min(axis=0)
        max_orig = origs.max(axis=0)
        reg = list(min_orig) + [x+y-z for x,y,z in zip(max_orig, self.adjacency_matrix.GetSpacing(), min_orig)]

        pixel_span = [int(x/z) for x,z in zip(reg[3:], self.spacing)]
        for index in indices:
            self.read_mosaic_piece(index)
        multi_image = sitk.Image(pixel_span, self.pieces[index].GetPixelID())
        multi_image.SetOrigin(min_orig)
        multi_image.SetSpacing(self.spacing)
        pasteFilter = sitk.PasteImageFilter()
        pasteFilter.SetSourceSize(self.pieces[index].GetSize())
        pasteFilter.SetSourceIndex([0, 0, 0])
        for index in indices:
            dest_indices = [int((x-y)/z) for x,z, y in zip(self.pieces[index].GetOrigin(), multi_image.GetSpacing(), multi_image.GetOrigin())]
            pasteFilter.SetDestinationIndex(dest_indices)
            multi_image = pasteFilter.Execute( multi_image, self.pieces[index])
            self.unload_mosaic_piece(index)

        return multi_image



    def process_subregion(self, fcn_hndle, region, spatial=True, append_string=''):
        if spatial:
            # convert region into voxel coordinates
            region = [float(region_param / resolution) for region_param, resolution in
                      zip(region, self.adjacency_matrix.GetSpacing() + self.adjacency_matrix.GetSpacing())]

        else:
            # convert pixels into macro space
            region = [float(region_param / size) for region_param, size in
                      zip(region, self.piece_size + self.piece_size)]

        # make sure that the subregion is contained within the selection of piece indices
        for i in range(0, 3):
            upshift = region[i] % 1
            region[i] = int(np.floor(region[i]))
            j = i + 3
            region[j] = int(np.ceil(region[j] + upshift))

        indices = self.get_piece_indices_for_sub_region(region)
        self.apply_function_over_pieces(fcn_hndle, indices, append_string=append_string + '_subregion')
        return indices

    def linearIndex_to_matrixIndices(self, indx):
        # slice_stack_number*grid_size**2 + i*grid_size**1 + j*grid_size**0

        z_indx = indx // (self.grid_size[0] * self.grid_size[1])
        indx = indx - indx // (self.grid_size[0] * self.grid_size[1]) * (self.grid_size[0] * self.grid_size[1])
        x_indx = indx // self.grid_size[1] ** 1
        indx = indx - indx // self.grid_size[1] ** 1 * self.grid_size[1] ** 1
        y_indx = indx // self.grid_size[1] ** 0

        return x_indx, y_indx, z_indx

    def visualise_slice(self, indx, axis='z'):
        """
        Parameters
        ----------
        @indx - this indx is the whole reconstruction scale index
        @axis - this parameter specifies the axis to which the plane being visualised is orthogonal to

        Returns
        -------
        :param indx:
        :param axis:

        """
        x_size, y_size, z_size = self.macro_size
        if axis == 'z':
            # test axis limits TO DO
            piece_number = self.location_of_point((0, 0, indx * self.spacing[2]))
            x_indx, y_indx, z_indx = self.linearIndex_to_matrixIndices(piece_number)
            piece_list = sitk.GetArrayFromImage(self.adjacency_matrix[:, :, z_indx]).flatten()

            mosaic_slice = sitk.Image([x_size, y_size], sitk.sitkUInt8)
            mosaic_slice.SetOrigin((0, 0))
            mosaic_slice.SetSpacing(self.spacing[:2])  # currently slice is purely 2D
            z_indx_local = indx - z_indx * self.piece_size[2]
            slice_size_local = self.piece_size[:2] + [0]  # the last zero signifies the collapse of the z axis
            extract_filter = partial(sitk.Extract, size=slice_size_local, index=[0, 0, z_indx_local])
            sub_imgs = self.apply_function_over_pieces(extract_filter, piece_list, external_output=False)
            paste_filter = sitk.PasteImageFilter()
            paste_filter.SetSourceIndex([0, 0])
            paste_filter.SetSourceSize(slice_size_local[:2])
            for img in sub_imgs:
                origin = img.GetOrigin()
                img_arr = sitk.GetArrayFromImage(img).flatten()
                # print(img_arr.min() == img_arr.max(), img_arr.min(), img_arr.max())
                destination = mosaic_slice.TransformPhysicalPointToIndex(origin)
                paste_filter.SetDestinationIndex(destination)
                mosaic_slice = paste_filter.Execute(mosaic_slice, img)
        elif axis == 'x':

            piece_number = self.location_of_point((indx * self.spacing[0], 0, 0))
            x_indx, y_indx, z_indx = self.linearIndex_to_matrixIndices(piece_number)
            piece_list = sitk.GetArrayFromImage(self.adjacency_matrix[x_indx - 1, :, :]).flatten()

            mosaic_slice = sitk.Image([y_size, z_size], sitk.sitkUInt8)
            mosaic_slice.SetOrigin((0, 0))
            mosaic_slice.SetSpacing(self.spacing[1:])  # currently slice is purely 2D
            x_indx_local = indx - 1 - (x_indx - 1) * self.piece_size[0]
            slice_size_local = [0] + self.piece_size[1:]  # the initial zero signifies the collapse of the x-axis
            extract_filter = partial(sitk.Extract, size=slice_size_local, index=[x_indx_local, 0, 0])
            sub_imgs = self.apply_function_over_pieces(extract_filter, piece_list, external_output=False)
            paste_filter = sitk.PasteImageFilter()
            paste_filter.SetSourceIndex([0, 0])
            paste_filter.SetSourceSize(slice_size_local[1:])
            for img in sub_imgs:
                origin = img.GetOrigin()
                img_arr = sitk.GetArrayFromImage(img).flatten()
                # print(img_arr.min() == img_arr.max(), img_arr.min(), img_arr.max())
                destination = mosaic_slice.TransformPhysicalPointToIndex(origin)
                paste_filter.SetDestinationIndex(destination)
                mosaic_slice = paste_filter.Execute(mosaic_slice, img)
        return mosaic_slice

    def reconstruct_macro_image(self, do_resample=False, isotropic_resample_scale=0.0):
        """
        This function
        Returns
        -------

        """
        if do_resample and isotropic_resample_scale != 0:
            resampled_spacing = np.array(self.spacing) * isotropic_resample_scale
            resampled_mosaic_piece_size = [int(x / isotropic_resample_scale + 0.5) for x in self.piece_size]
            resample = sitk.ResampleImageFilter()
            resample.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # Identity matrix
            resample.SetOutputSpacing(resampled_spacing)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            resample.SetSize(resampled_mosaic_piece_size)
            resampled_img_size = [int(x / isotropic_resample_scale) for x in self.macro_size]
            rss_x, rss_y, rss_z = resampled_img_size
            macro_img = sitk.Image(resampled_img_size, sitk.sitkUInt8)
            macro_img.SetSpacing(resampled_spacing)
            z_stack = sitk.Image([rss_x, rss_y, self.macro_size[2]], sitk.sitkUInt8)
            z_stack.SetSpacing([resampled_spacing[0], resampled_spacing[1], self.spacing[2]])

        else:
            macro_img = sitk.Image(self.macro_size, sitk.sitkUInt8)
            macro_img.SetSpacing(self.spacing)
        paste_filter = sitk.PasteImageFilter()
        paste_filter.SetSourceIndex([0, 0, 0])

        for indx in range(0, self.grid_size[0] * self.grid_size[1] * self.grid_size[2]):
            err = 0
            if indx not in self.pieces.keys():
                err = self.read_mosaic_piece(indx)

            print(f"\rread in image {indx}", end='')
            if do_resample:

                if indx % self.grid_size[1] == 0:
                    # create temporary image in the y direction
                    temp_y = sitk.Image([self.piece_size[0], self.macro_size[1], self.piece_size[2]],
                                        sitk.sitkUInt8)
                    temp_y.SetSpacing(self.spacing)
                    temp_y.SetOrigin(self.elements[indx])
                    if indx % (self.grid_size[0] * self.grid_size[1]) == 0:
                        # create temp layer image
                        temp_layer = sitk.Image([self.macro_size[0], rss_y, self.piece_size[2]], sitk.sitkUInt8)
                        temp_layer.SetOrigin(self.elements[indx])
                        temp_layer.SetSpacing([self.spacing[0], resampled_spacing[1], self.spacing[2]])
                if err == 0:
                    mosaic_piece = self.pieces[indx]
                    origin = self.elements[indx]
                    destination = temp_y.TransformPhysicalPointToIndex(origin)
                    paste_filter.SetDestinationIndex(destination)

                    resample.SetOutputOrigin(origin)
                    paste_filter.SetSourceSize(self.piece_size)
                    temp_y = paste_filter.Execute(temp_y, mosaic_piece)

                if indx % self.grid_size[1] == self.grid_size[1] - 1:
                    # temp_y is now fully stacked with images
                    resample.SetSize([temp_y.GetSize()[0], rss_y, temp_y.GetSize()[2]])
                    resample.SetOutputSpacing(temp_layer.GetSpacing())
                    resample.SetOutputOrigin(temp_y.GetOrigin())
                    resampled_y_stack = resample.Execute(temp_y)

                    # paste resampled stack into temporary layer image
                    destination = temp_layer.TransformPhysicalPointToIndex(temp_y.GetOrigin())
                    paste_filter.SetSourceSize(resampled_y_stack.GetSize())
                    paste_filter.SetDestinationIndex(destination)
                    temp_layer = paste_filter.Execute(temp_layer, resampled_y_stack)

                if indx % (self.grid_size[0] * self.grid_size[1]) == (self.grid_size[0] * self.grid_size[1]) - 1:
                    # oooh yay we have a complete layer
                    resample.SetSize([rss_x, rss_y, self.macro_size[2]])
                    resample.SetOutputSpacing(z_stack.GetSpacing())
                    resample.SetOutputOrigin(temp_layer.GetOrigin())
                    resampled_layer = resample.Execute(temp_layer)

                    destination = z_stack.TransformPhysicalPointToIndex(temp_layer.GetOrigin())
                    paste_filter.SetSourceSize(resampled_layer.GetSize())
                    paste_filter.SetDestinationIndex(destination)
                    z_stack = paste_filter.Execute(z_stack, resampled_layer)
            else:
                origin = self.elements[indx]
                destination = macro_img.TransformPhysicalPointToIndex(origin)
                paste_filter.SetDestinationIndex(destination)
                paste_filter.SetSourceSize(self.pieces[indx].GetSize())
                macro_img = paste_filter.Execute(macro_img, self.pieces[indx])
            if err == 0:
                self.unload_mosaic_piece(indx)
        # print(f"Mosaic piece {indx} successfully processed")
        print("\n")
        if do_resample:
            resample.SetSize([rss_x, rss_y, rss_z])
            resample.SetOutputSpacing(resampled_spacing)
            resample.SetOutputOrigin(z_stack.GetOrigin())
            resampled_img = resample.Execute(z_stack)

            print("And here we are once again")

        return resampled_img

    def remove_blank_pieces(self, upperLimit=0.0):

        for indx in self.elements.keys():

            err = self.read_mosaic_piece(indx)
            if err == 0:
                flat_im = sitk.GetArrayViewFromImage(self.pieces[indx])
                max_val = flat_im.max()
                if max_val < upperLimit:
                    self.remove_mosaic_piece(indx)
                self.unload_mosaic_piece(indx)
            else:
                pass
        return

    def remove_mosaic_piece(self, indx):
        file_list = os.listdir(self.directory)
        file_name = ""
        for x in file_list:
            if x.split(sep='_', maxsplit=1)[0] == str(indx):
                file_name = x
                break
        if file_name != "":
            os.remove(os.path.join(self.directory, file_name))
            return 0
        else:
            return
