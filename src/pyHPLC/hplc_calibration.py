# -*- coding: utf-8 -*-
"""
Contains class hplc_calibration.

Can only be used in combination with hplc_data and makes most sense in
combination with hplc_prediction.
"""

import numpy as np
import pandas as pd

from pyHPLC import hplc_data
from pyDataFitting.multivariate_regression import (
    principal_component_regression, pls_regression)


class hplc_calibration():
    """
    Class holding and analyzing HPLC calibration measurements.

    Contains one calibration set of one substance per instance and analyzes
    them by classical least squares regresseion or principal component
    regression.
    """

    def __init__(self, data_mode, calibration_data, concentrations,
                 time_limits=None, wavelength_limits=None,
                 internal_standard=False, **kwargs):
        """
        Use input data and store them internally in hplc_data instances.

        Parameters
        ----------
        data_mode : string
            Defines the way the data is imported into the instance. At the
            moment, input as instances of hplc_data or direct input of
            DataFrames with the wavelengths as columns and the elution time as
            index is possible, so the allowed values are 'hplc_data' or
            'DataFrame'. Might be extended in the future for direct data import
            from files.
        calibration_data : list of DataFrames or hplc_object instances
            Contains the data from the n calibration measurements either as
            DataFrames with the wavelengths as columns and the elution time as
            index or as instances of hplc_data objects (depending on
            data_mode).
        concentrations : list
            Contains the n concentrations present in the n calibration
            measurements contained in calibration_data in the same order.
        time_limits : list or None, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams used for calibration. Each element can
            either be a number or None. In the latter case, no data is removed
            from the respective side. The default is None.
        wavelength_limits : list or None, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra used for calibration. Each
            element can either be a number or None. In the latter case, no data
            is removed from the respective side. The default is None.
        internal_standard : bool, optional
            Defines if an internal standard was used (True) or not (False). 
            If this is True, the calibration_data will be normalized
            accordingly. The default is False.
        **kwargs : 
            pcr_components : int
                Determines the number of principal components used for
                principal component regression calibration. The default is 2.
            is_time_limits: list
                A list containing two elements: The start time in the elugrams
                and the end time in the elugrams used for internal standard. 
                Only used when internal_standard is True. The default is None.
            is_wavelength_limits: list
                A list containing two elements: The start wavelength  and the
                end wavelength used for internal standard. Only used when
                internal_standard is True. The default is None.
        Raises
        ------
        ValueError
            If no valid data_mode is given.

        Returns
        -------
        None.

        """
        self.concentrations = np.array(concentrations)
        self.internal_standard = internal_standard

        if data_mode == 'DataFrame':
            self.calibration_data_raw = []
            for curr_data in calibration_data:
                self.calibration_data_raw.append(
                    hplc_data('DataFrame', data=curr_data))
        elif data_mode == 'hplc_data':
            self.calibration_data_raw = calibration_data
        else:
            raise ValueError('No valid data_mode given.')

        self.set_limits(time_limits, wavelength_limits)

        if self.internal_standard:
            self.is_time_limits = kwargs.get('is_time_limits', None)
            self.is_wavelength_limits = kwargs.get('is_wavelength_limits', None)
            self.calibration_data = []
            for curr_data in self.calibration_data_raw:
                self.calibration_data.append(hplc_data(
                    'DataFrame', data=
                    curr_data.normalize(
                        'internal_standard',
                        time_limits=self.is_time_limits,
                        wavelength_limits=self.is_wavelength_limits)))
        else:
            self.calibration_data = self.calibration_data_raw.copy()

        self.integrate_calibration_data()
        self.classical_least_squares()

        # Principal component regression can only be performed if more than one
        # wavelength is used for calibration.
        if len(np.squeeze(self.calibration_integrated).shape) > 1:
            pcr_components = kwargs.get('pcr_components', 2)
            plsr_components = kwargs.get('plsr_components', 2)
            self.principal_component_regression(
                n_components=pcr_components,
                cv_percentage=1/len(self.calibration_data)*100)
            self.pls_regression(n_components=plsr_components,
                                cv_percentage=1/len(self.calibration_data)*100)

    def classical_least_squares(self):
        """
        Calculate the calibration by classical least squares fitting.

        The result is stored in self.K. At the moment, a self implemented
        solution is used which might be replaced by sklearn.linear_model in the
        future (like e.g. used in principal component regression).

        Returns
        -------
        DataFrame
            The calibration result containing the slopes at the different
            wavelengths obtained by classical least squares fitting.

        """
        cls_concentrations = self.concentrations[np.newaxis]

        self.K = np.dot(
            np.dot(
                self.calibration_integrated, cls_concentrations.T
                ),
            np.linalg.inv(
                np.dot(
                    cls_concentrations,
                    cls_concentrations.T)
                ))
        self.K = pd.DataFrame(self.K, index=self.calibration_integrated.index)

        return self.K

    def principal_component_regression(self, n_components=2, cv_percentage=20):
        """
        Perform calibration based on principal component regression.

        The calibration is represented by an instance of
        principal_component_regression and thus already holds a prediction
        method.

        Parameters
        ----------
        n_components : int, optional
            The number of principal components used for the multilinear
            regression. The default is 2.
        cv_percentage : float, optional
            Percentage of the data to be used for cross validation. Must be at
            least that big that it includes one sample. The default is 20.

        Returns
        -------
        None.

        """
        self.pcr_calibration = principal_component_regression(
            self.calibration_integrated.values.T, self.concentrations)
        self.pcr_components = n_components
        self.pcr_calibration.pcr_fit(self.pcr_components,
                                     cv_percentage=cv_percentage)

    def pls_regression(self, n_components=2, cv_percentage=20, scale=False):
        self.plsr_calibration = pls_regression(
            self.calibration_integrated.values.T, self.concentrations)
        self.plsr_components = n_components
        self.plsr_calibration.plsr_fit(self.plsr_components,
                                       cv_percentage=cv_percentage,
                                       scale=scale)

    def integrate_calibration_data(self):
        """
        Integrate all calibration datasets within the limits given.

        Returns
        -------
        None.

        """
        self.calibration_integrated = []
        self.calibration_raw_integrated = []
        for curr_data, curr_raw_data in zip(
                self.calibration_data, self.calibration_data_raw):
            self.calibration_integrated.append(
                curr_data.integrate_all_data(
                    time_limits=self.time_limits,
                    wavelength_limits=self.wavelength_limits))

            self.calibration_raw_integrated.append(
                curr_raw_data.integrate_all_data(
                    time_limits=self.time_limits,
                    wavelength_limits=self.wavelength_limits))

        self.calibration_integrated = pd.DataFrame(
            self.calibration_integrated).T
        self.calibration_raw_integrated = pd.DataFrame(
            self.calibration_raw_integrated).T

    def set_limits(self, time_limits, wavelength_limits):
        """
        Save the time and wavelength limits to the respective attributes.

        Parameters
        ----------
        time_limits : list or None, optional
            A list containing two elements: The start time in the elugrams and
            the end time in the elugrams used for calibration. Each element can
            either be a number or None. In the latter case, no data is removed
            from the respective side. The default is None.
        wavelength_limits : list or None, optional
            A list containing two elements: The start wavelength in the spectra
            and the end wavelength in the spectra used for calibration. Each
            element can either be a number or None. In the latter case, no data
            is removed from the respective side. The default is None.

        Returns
        -------
        None.

        """
        if time_limits is not None:
            self.time_limits = time_limits
        else:
            self.time_limits = None
        if wavelength_limits is not None:
            self.wavelength_limits = wavelength_limits
        else:
            self.wavelength_limits=None

    # currently only works for single wavelength calibration
    def generate_report(self):
        report_data = pd.DataFrame([], columns=['concentration',
                                                'raw peak area',
                                                'internal standard area',
                                                'normalized peak area'])
        report_data['concentration'] = self.concentrations
        report_data['raw peak area'] = np.squeeze(
            self.calibration_raw_integrated.values)
        report_data['normalized peak area'] = np.squeeze(
            self.calibration_integrated.values)
        if self.internal_standard:
            for ii, curr_data in enumerate(self.calibration_data_raw):
                report_data.loc[ii,'internal standard area'] = curr_data.standard_data.values
        
        return report_data