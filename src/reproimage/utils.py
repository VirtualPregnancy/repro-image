import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import sqlite3
import pandas as pd
import copy
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

    all_trough_indices, _ = find_peaks(-1*y_values)  # Negate y_values to find minima
    peak_indices, _ = find_peaks(y_values, prominence=wave_amplitude / 4)  # Negate y_values to find minima
    ## Want to find the last local minima that has occured before a main peak
    trough_indices = []
    for peak in peak_indices:
        trough_loc = np.where(all_trough_indices<peak)[0]
        trough_loc = trough_loc[-1]
        trough_indices.append(all_trough_indices[trough_loc])


    interpolated_waves = []
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label="Waveform", color='blue')
        plt.scatter(x_values[all_trough_indices], y_values[all_trough_indices], color='blue', label='Troughs', zorder=5)
        plt.scatter(x_values[peak_indices], y_values[peak_indices], color='black', label='Peaks', zorder=5)
        plt.scatter(x_values[trough_indices], y_values[trough_indices], color='red', label='Final Troughs', zorder=5)

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

    # Convert the list of interpolated waves to a NumPy array for calculations
    interpolated_waves_np = np.vstack(interpolated_waves)

    # Calculate the initial average and standard deviation
    average_wave = np.mean(interpolated_waves_np, axis=0)
    std_wave = np.std(interpolated_waves_np, axis=0)
    amplitude_of_ave = np.max(average_wave)-np.min(average_wave)

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.title("Set of Waveforms")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        for wave_index, waveform in enumerate(interpolated_waves):
            plt.plot(x_common, waveform, label=f"Waveform {wave_index}")
        plt.plot(x_common, average_wave, label='Average wave', linestyle='-.')
        plt.plot(x_common, average_wave+std_wave, label='Average wave + SD', linestyle='--')
        plt.plot(x_common, average_wave-std_wave, label='Average wave - SD', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Filter out waves outside the range of average ± standard deviation
    threshold_percentage = 80

    filtered_waves = []
    excluded_waves = []
    count_excluded = 0
    for wave in interpolated_waves_np:
        # Calculate the percentage of points that meet the OR condition
        within_range = (wave >= (average_wave - 0.2*amplitude_of_ave)) & (
                    wave <= (average_wave + 0.2*amplitude_of_ave))  # Points above or equal to lower bound
        percentage_within_range = np.sum(within_range) / len(wave) * 100
        # Include the wave if the percentage is above the threshold
        if percentage_within_range >= threshold_percentage:
            filtered_waves.append(wave)
        else:
            count_excluded =+ 1
            excluded_waves.append(wave)
    if verbose:
        print("Waves filtered, num excluded", count_excluded)




    if verbose:
        print(f"{len(interpolated_waves)} waveforms included in the calculationg for the average waveform,"
              f"using a cutoff proportion of {threshold_percentage} % for points within one standard deviation of the "
              f"raw native waveform")
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
        #plt.plot(x_common, excluded_waves[0])
        plt.grid(True)
        plt.show()

    return new_average_wave, x_common


class MetaData:
    def __init__(self, dataframe=None, database_path=None):
        self.dataframe = copy.deepcopy(dataframe)
        if (dataframe is not None) ^ bool(database_path):
            if database_path:
                self.dataframe = copy.deepcopy(self.__parse_metadata_db(database_path))

            assert self.dataframe.index.is_unique, (
                "Dataframe indexes must be unique, there is a risk using the truncated hash "
                "values but the risk is small")
        else:
            print("INITIALISATION ERROR, user should provide ONE of <dataframe> or <path to a sql database>\n"
                  "NOT both or neither")

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls(dataframe=dataframe)

    @classmethod
    def from_db_path(cls, db_path):
        return cls(database_path=db_path)

    def __process_row(self, row):
        row = list(row)
        row[0] = int(row[0].split('-')[-1])  # truncate Daphne-<number> to just <number>
        row[5] = row[5][:8]  # truncate the image hash to the first 8 characters
        return row

    def __parse_metadata_db(self, sql_database):
        assert sql_database.exists(), "Invalid database provided, it does not exist"
        # read db into RAM as pandas dataframe
        con = sqlite3.connect(sql_database)
        cur = con.cursor()
        res = cur.execute("PRAGMA table_info(daphne_metadata)")
        cols = [x[1] for x in res.fetchall()]
        col_str = ','.join(cols)
        res = cur.execute(f"SELECT {col_str} FROM daphne_metadata")
        metadata = res.fetchall()
        con.close()

        cols[0] = 'Daphne Number'
        metadata = [self.__process_row(x) for x in metadata]
        dataframe = pd.DataFrame(metadata, columns=cols).set_index('image_hash')
        pd.options.display.max_columns = len(cols)
        dataframe[cols[7:]] = dataframe[cols[7:]].apply(pd.to_numeric, errors='coerce')
        dataframe['HR'] = dataframe['HR'].astype(int)
        dataframe['datetime'] = pd.to_datetime(dataframe['dateofexam'],
                                                                 format='%Y%m%d')
        # drop columns with no unique data
        for col in dataframe.columns:
            if dataframe[col].unique().size == 1:
                print(f"Dropping {col} as all rows have the value: {dataframe[col].unique()[0]}")
                dataframe.drop(col, axis=1, inplace=True)

        return dataframe

    def __get__column(self, col):
        return self.dataframe[col]

    def summary_data(self):
        print(self.dataframe.describe(include='all'))

    def mean(self, column):
        print(self.__get__column(column).mean())

    def Get_Fields(self, cols):
        return MetaData.from_dataframe(self.dataframe[cols])

    def Get_Field_Names(self):
        fields = self.dataframe.columns.to_list()
        return fields

    def Get_Directories(self):
        dirs = {}
        for daphne_number in self.dataframe['Daphne Number'].unique():
            image_dir = f"Primary/leap/doppler/Daphne-{daphne_number}"
            waveform_dir = f"Derivative/leap/doppler/Daphne-{daphne_number}"
            dirs[daphne_number] = {'image directory': image_dir, "waveform directory": waveform_dir}
        return dirs

    def Get_Daphnes(self, daphnes):
        return MetaData.from_dataframe(self.dataframe[self.dataframe['Daphne Number'].isin(daphnes)])

    def __get__uniques(self, series):
        return series.unique()

    def Get_Vessels(self):
        series = self.__get__column('prefix')
        return self.__get__uniques(series).tolist()

    def Get_Index_as_array(self):
        return self.dataframe.index.to_numpy()

    def Get_Num_Entries(self):
        return self.dataframe.shape[0]
    def Query(self, query, local_dictionary=None):
        try:
            if local_dictionary:
                subset = self.dataframe.query(query, local_dict = local_dictionary)
            else:
                subset = self.dataframe.query(query)
            return MetaData.from_dataframe(subset)
        except:
            print("Error executing your query, sql like queries are accepted e.g:\n"
                  "`Daphne Number` == 44 and HR < 120 or\n"
                  "PI > 1"
                  "prefix == \"LT\""
                  "You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b."
                  "You can refer to column names that are not valid Python variable names by surrounding them in backticks.")
        return
    def Get_Earliest_Scan_Datetime(self):
        return self.dataframe['datetime'].min()

    def Get_Latest_Scan_Datetime(self):
        return self.dataframe['datetime'].max()

    def Sort(self, field, ascending=True):
        """Sort the database by the values in the <field>, ascending by default controlled by the <ascending> parameter,
        this happens in place"""
        self.dataframe.sort_values(field, axis='columns', ascending=ascending)
    def Summary(self, verbose=False):
        num_images = self.Get_Num_Entries()
        fields = self.Get_Field_Names()
        if 'Daphne Number' in fields:
            unique_patients = self.dataframe['Daphne Number'].unique().size
            print(f"This data containts information from {unique_patients} distinct patient(s)")
        num_fields = len(fields)
        print(
            f"This data contains information from {num_images} unique images, each with {num_fields} fields, {fields}")
        if 'prefix' in fields:
            vessels = self.Get_Vessels()
            if 'N/A' in vessels:
                unknown_index = vessels.index('N/A')
                print(
                    f"There are {len(vessels) - 1} distinct vessesl that are imaged in this dataset, {[x for x in vessels if x != 'N/A']},including some that have "
                    f"not been specified, these are labelled {vessels[unknown_index]}")
            else:
                print(f"There are {len(vessels)} distinct vessesl, {vessels} that are imaged in this dataset")
        if verbose:
            self.summary_data()

    def Get_Image_Data(self, im_hash):
        """"accepts either singular argumebts specifying a single image index, or a list of image indices"""
        return self.dataframe.loc[im_hash]

    def Generate_timeDelta(self, delta_key, delta_value):
        if delta_key in ['day', 'days', 'month', 'months', 'year', 'years']:
            return pd.Timedelta(delta_value, unit=delta_key)
        else:
            print("USEAGE ERROR: correct useage is Generate_timeDelta(time_units, time_value\n"
                  "Your units must be one of 'day', 'days', 'month', 'months', 'year', 'years'")

    def to_numpy(self):
        return self.dataframe.to_numpy()