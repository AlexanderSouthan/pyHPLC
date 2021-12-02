# -*- coding: utf-8 -*-
"""
Contains class hplc_data.

Best used in combination with hplc_calibration and hplc_prediction.
"""

import numpy as np
import pandas as pd

from pyAnalytics.measurement_parameters import measurement_parameters
from pyPreprocessing.baseline_correction import generate_baseline


class hplc_data():
    """
    Class holding 3D HPLC data with methods for data treatment.

    Contains one measurement of 3D HPLC data per instance.
    """

    def __init__(self, mode, **kwargs):
        """
        Load HPLC data.

        Parameters
        ----------
        mode : str
            Defines the method of data handed over. If 'import': Data is
            imported from 3D data files exported by Shimadzu LCsolution
            software. If 'DataFrame': Data is a pandas DataFrame with elution
            times as index and wavelengths as columns. See also necessary
            **kwargs for the different modes.
        **kwargs for mode == 'import':
            file_name : str
                File name of file containing the measurement data.
            directory : str
                Directory containing the file given in file_name. Note that it
                must be given without a trailing '/'.
            full_path : str
                An alternative to the separate file_name and directory, is the
                full path to the imported file. If a value is given, it is
                treated with a higher priority than file_name and directory.
        **kwargs for mode == 'DataFrame':
            data : DataFrame
                Data is a pandas DataFrame with elution times as index and
                wavelengths as columns.

        Raises
        ------
        ValueError
            If no valid mode is given.

        Returns
        -------
        None.

        """
        if mode == 'import':
            self.import_path = kwargs.get('full_path', None)
            
            if self.import_path is None:
                self.file_name = kwargs.get('file_name')
                directory = kwargs.get('directory', None)
                if directory is None:
                    self.directory = ''
                else:
                    self.directory = directory + '/'
                self.import_path = self.directory + self.file_name

            self.import_from_file(self.import_path)
        elif mode == 'DataFrame':
            self.raw_data = kwargs.get('data')
            self.wavelengths = self.raw_data.columns.to_numpy()
            measurement_params = pd.DataFrame(
                [self.raw_data.index[1] - self.raw_data.index[0],
                 self.raw_data.index[0], self.raw_data.index[-1],
                 self.raw_data.columns[0], self.raw_data.columns[-1],
                 len(self.raw_data.columns), len(self.raw_data.index)],
                index=['interval', 'start_time', 'end_time',
                       'start_wavelength', 'end_wavelength',
                       'number_of_wavelength_points',
                       'number_of_time_points'])
            self.PDA_data = measurement_parameters(measurement_params)
        else:
            raise ValueError('No valid mode entered. Allowed modes are' +
                             ' \'import\' and \'DataFrame\'.')

        self.time_data = self.raw_data.index.to_numpy()
        self.processed_data = self.raw_data.copy()
        self.baseline_data = {}

    def import_from_file(self, import_path):
        """
        Import one measurement's 3D HPLC data from Shimadzu LabSolution.

        Is only called if mode in __init__ is 'import'. Stores the data in
        self.raw_data.

        Parameters
        ----------
        import path : str
            The full path including the file name for import.

        Returns
        -------
        None.

        """
        # next line in case function is called after mode was not 'import'
        self.import_path = import_path

        PDA_data_raw = pd.read_csv(self.import_path, sep='\t',
                                   skiprows=13, nrows=7)
        self.PDA_data = measurement_parameters(
            PDA_data_raw, parameter_names=[
                'interval', 'start_time', 'end_time', 'start_wavelength',
                'end_wavelength', 'number_of_wavelength_points',
                'number_of_time_points'])
        self.PDA_data.number_of_wavelength_points = int(
            self.PDA_data.number_of_wavelength_points)
        self.PDA_data.number_of_time_points = int(
            self.PDA_data.number_of_time_points)

        self.wavelengths = np.linspace(
            self.PDA_data.start_wavelength, self.PDA_data.end_wavelength,
            num=self.PDA_data.number_of_wavelength_points)

        self.raw_data = np.loadtxt(self.import_path,
                                   skiprows=23, delimiter='\t')
        self.raw_data = pd.DataFrame(self.raw_data)
        self.raw_data.set_index(0, inplace=True)
        self.raw_data.columns = self.wavelengths

    def crop_data(self, time_limits=None, wavelength_limits=None,
                  active_data=None):
        """
        Use to extract one part from the complete 3D HPLC datset self.raw_data.

        Parameters
        ----------
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams, defining the data range to be kept
            after processing. Each element can either be a number or None. In
            the latter case, no data is removed from the respective side. The
            default is None.
        wavelength_limits : list, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra defining the data range to be
            kept after processing. Each element can either be a number or None.
            In the latter case, no data is removed from the respective side.
            The default is None.
        active_data : DataFrame, optional
            Operations are done by default on self.raw_data. If other
            preprocessing must be done prior to calling this method, this
            preprocessed data in a similar format as self.raw_data can be
            passed as this argument. The default is None.

        Returns
        -------
        cropped_data : TYPE
            DESCRIPTION.

        """
        active_data = self.check_active_data(active_data)

        if time_limits is None:
            time_limits = [self.PDA_data.start_time, self.PDA_data.end_time]
        elif None in time_limits:
            if time_limits[0] is None:
                time_limits[0] = self.PDA_data.start_time
            if time_limits[1] is None:
                time_limits[1] = self.PDA_data.end_time

        if wavelength_limits is None:
            wavelength_limits = [
                self.PDA_data.start_wavelength, self.PDA_data.end_wavelength]
        elif None in wavelength_limits:
            if wavelength_limits[0] is None:
                wavelength_limits[0] = self.PDA_data.start_wavelength
            if wavelength_limits[1] is None:
                wavelength_limits[1] = self.PDA_data.end_wavelength

        cropped_data = active_data.iloc[
            self.closest_index_to_value(
                active_data.index, time_limits[0]):
            self.closest_index_to_value(
                active_data.index, time_limits[1]) + 1,
            self.closest_index_to_value(
                active_data.columns, wavelength_limits[0]):
            self.closest_index_to_value(
                active_data.columns, wavelength_limits[1]) + 1
            ]

        self.processed_data = cropped_data

        return cropped_data

    def extract_elugram(self, wavelength, time_limits=None,
                        active_data=None):  # , baseline_correction=False):
        """
        Extract a single elugram from one wavelength from the 3D HPLC dataset.

        Parameters
        ----------
        wavelength : float
            The wavelength at which the elugram is extracted from 3D dataset.
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams, defining the data range to be kept
            after processing. Each element can either be a number or None. In
            the latter case, no data is removed from the respective side. The
            default is None.
        active_data : DataFrame or None, optional
            Data to be processed by the method if not None. If None, the
            processed data is self.raw_data. The default is None.

        Returns
        -------
        elugram : DataFrame
            A DataFrame with one column containing the elugram and the time as
            index.

        """
        active_data = self.check_active_data(active_data)

        elugram = self.crop_data(
            time_limits=time_limits,
            wavelength_limits=[wavelength, wavelength], active_data=active_data
            )

        # if baseline_correction:
        #     # baseline is not saved in any form and so far no method for full
        #     # baseline correction exists
        #     baseline = pd.Series(
        #         pyPreprocessing.baseline_correction.drPLS_baseline(
        #             elugram.values[np.newaxis,:],100000000,0.8,100)[0,:].T,
        #             index = elugram.index)
        #     elugram = elugram - baseline

        return elugram

    def extract_spectrum(self, elution_time, wavelength_limits=None,
                         active_data=None):
        active_data = self.check_active_data(active_data)

        spectrum = self.crop_data(
            time_limits=[elution_time, elution_time],
            wavelength_limits=wavelength_limits, active_data=active_data
            ).T

        return spectrum

    def integrate_elugram(self, wavelength, time_limits=None,
                          active_data=None):  # , baseline_correction=False):
        """
        Integrate a single elugram from one wavelength along the time axis.

        Parameters
        ----------
        wavelength : float
            The wavelength at which the elugram is extracted from 3D dataset.
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams, defining the data range to be kept
            after processing. Each element can either be a number or None. In
            the latter case, no data is removed from the respective side. The
            default is None.
        active_data : DataFrame or None, optional
            Data to be processed by the method if not None. If None, the
            processed data is self.raw_data. The default is None.

        Returns
        -------
        elugram_integrated : float
            The integration result.

        """
        active_data = self.check_active_data(active_data)

        curr_elugram = self.extract_elugram(
            wavelength, time_limits=time_limits,
            active_data=active_data)
        # , baseline_correction=baseline_correction)
        elugram_integrated = np.trapz(curr_elugram, x=curr_elugram.index,
                                      axis=0).item()

        return elugram_integrated

    def generate_projections(self, dim='time', time_limits=None,
                             wavelength_limits=None, active_data=None):
        active_data = self.check_active_data(active_data)

        active_data = self.crop_data(time_limits=time_limits,
                                     wavelength_limits=wavelength_limits,
                                     active_data=active_data)

        assert dim in ['time', 'wavelength'], 'Allowed values for dim are time and wavelength, current value is {}.'.format(dim)
        projection_axis = 1 if dim=='time' else 0
        column_labels = active_data.index if dim=='time' else active_data.columns

        projections = pd.DataFrame([
            np.amax(active_data.values, axis=projection_axis),
            np.amin(active_data.values, axis=projection_axis),
            np.mean(active_data.values, axis=projection_axis)],
            columns=column_labels, index=['max', 'min', 'mean']).T

        return projections

    def integrate_all_data(self, mode='elugrams', time_limits=None,
                           wavelength_limits=None, active_data=None):
        """
        Call to integrate 3D HPLC data along wavelength or time axis.

        Input data has N wavelengths and M time points.

        Parameters
        ----------
        mode : str, optional
            Determines the integration axis. If mode == 'elugrams', all
            elugrams are integrated along the time axis. If mode == 'spectra',
            all spectra are integrated along the wavelength axis. The default
            is 'elugrams'.
        time_limits : list, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams, defining the data range to be kept
            after processing. Each element can either be a number or None. In
            the latter case, no data is removed from the respective side. The
            default is None.
        wavelength_limits : list, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra defining the data range to be
            kept after processing. Each element can either be a number or None.
            In the latter case, no data is removed from the respective side.
            The default is None.
        active_data : DataFrame or None, optional
            Data to be processed by the method if not None. If None, the
            processed data is self.raw_data. The default is None.

        Returns
        -------
        integrated_data : Series
            The integrated data. If mode == 'elugram', it contains N elements.
            If mode == 'spectra', it contains M elements.

        """
        # baseline_correction not yet integrated
        active_data = self.check_active_data(active_data)
        active_data = self.crop_data(time_limits=time_limits,
                                     wavelength_limits=wavelength_limits,
                                     active_data=active_data)

        if mode == 'elugrams':
            integration_axis = 0
            x_data = active_data.index
            result_index = active_data.columns
        elif mode == 'spectra':
            integration_axis = 1
            x_data = active_data.columns
            result_index = active_data.index

        integrated_data = np.trapz(active_data.values, x=x_data,
                                   axis=integration_axis)
        integrated_data = pd.Series(integrated_data,
                                    index=result_index)
        return integrated_data

    def normalize(self, mode, active_data=None, **kwargs):
        active_data = self.check_active_data(active_data)

        normalization_modes = ['internal_standard']

        if mode == normalization_modes[0]:
            time_limits = kwargs.get('time_limits', None)
            wavelength_limits = kwargs.get('wavelength_limits', None)

            active_data = self.crop_data(
                time_limits=None,
                wavelength_limits=wavelength_limits,
                active_data=active_data)

            self.standard_data = self.integrate_all_data(
                time_limits=time_limits, wavelength_limits=wavelength_limits,
                active_data=active_data).sum()

            normalized_data = active_data / self.standard_data
        else:
            raise ValueError('No valid normalization mode entered.')

        return normalized_data

    # next function should not stay in object
    def closest_index_to_value(self, array, value):
        """
        Call to find an array index representing a value closest to value.

        Parameters
        ----------
        array : ndarray
            The input array.
        value : float
            The value to be approximated.

        Returns
        -------
        int
            The index of the value in array closest to value.

        """
        return np.argmin(np.abs(array - value))

    def baseline_correction(self, elugrams='all', mode='ModPoly',
                            smoothing=True, transform=False, active_data=None,
                            **kwargs):
        """
        Correct baseline with methods from pyPreprocessing.baseline_correction.

        The details about the function arguments and kwargs can be found in the
        docstrings in pyPreprocessing.baseline_correction.

        Parameters
        ----------
        elugrams : str or float, optional
            Defines which elugrams are corrected. Default is 'all' meaning all
            elugrams at all wavelengths will be corrected, consuming the most
            computing time. Alternatively, a wavelength value can be pssed and
            only the elugram at that wavelength is corrected.
        mode : str, optional
            The baseline correction algorithm. Default is 'ModPoly'.
        smoothing : bool, optional
            True means the data are smoothed before baseline correction. The
            default is True.
        transform : bool, optional
            True means the data are transformed before baseline correction.
            The default is False.
        active_data : pandas DataFrame or None, optional
            None means that self.raw_data is used for baseline correction.
            Alternatively, a pandas DataFrame in the same format as
            self.raw_data can be passed that is used for the calculations.
            The default is None.
        **kwargs
            All kwargs necessary for the respective baseline correction mode.
            If x values are needed for the calculations, these do not need
            to be given because they are already known. If they are passed
            anyway, the passed values are ignored.

        Returns
        -------
        pd.DataFrame
            The spectral data with the subtracted baseline.

        """
        active_data = self.check_active_data(active_data)

        if elugrams == 'all':
            active_data_baseline = active_data
        else:
            active_data_baseline = self.extract_elugram(
                elugrams, active_data=active_data)

        active_data_baseline = active_data_baseline.T

        if mode in ['convex_hull', 'ModPoly', 'IModPoly', 'PPF', 'iALSS']:
            kwargs['wavenumbers'] = active_data_baseline.columns.to_numpy()

        curr_baseline_data = pd.DataFrame(
            generate_baseline(active_data_baseline.values, mode,
                              smoothing=True, transform=False, **kwargs),
            index=active_data_baseline.index,
            columns=active_data_baseline.columns)

        if elugrams == 'all':
            self.baseline_data[mode] = curr_baseline_data.T
        else:
            curr_index = self.closest_index_to_value(active_data.columns,
                                                     elugrams)
            self.baseline_data[mode] = pd.DataFrame(
                np.zeros_like(active_data.values), index=active_data.index,
                columns=active_data.columns)
            self.baseline_data[mode].iloc[
                :, curr_index] = np.squeeze(curr_baseline_data)

        corrected_data = (active_data -
                          self.baseline_data[mode]).round(decimals=6)

        self.processed_data = corrected_data

        return corrected_data

    def check_active_data(self, active_data):
        """
        Call to determine if active_data is None.

        Called by data processing methods to determine the dataset used for
        data processing. If active_data is None, the full raw dataset will be
        returned.

        Parameters
        ----------
        active_data : DataFrame or None
            Contains a DataFrame if data is passed to the calling method as
            active_data.

        Returns
        -------
        active_data : DataFrame
            Data to be processed by the calling method.

        """
        if active_data is None:
            active_data = self.processed_data
        return active_data
