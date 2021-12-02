# -*- coding: utf-8 -*-
"""
Contains class hplc_prediction.

Can only be used in combination with hplc_data and hplc_calibration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyDataFitting.nonlinear_regression import calc_function
from pyAnalytics.hplc_data import hplc_data
from pyAnalytics.hplc_calibration import hplc_calibration


class hplc_prediction():
    """
    Contains prediction methods for 3D HPLC data.

    3D HPLC data are time resolved absorption spectra, e.g. measured with a DAD
    detector. The underlying calibration can be univariate or multivariate
    based on classical least squares (multi)linear regression or principal
    component regression.
    """

    def __init__(self, samples, calibrations):
        """
        Only collects the data needed for the different prediction methods.

        Apart from that does nothing.

        Parameters
        ----------
        samples : list of hplc_data instances
            List of complete datasets from the HPLC measurements, collected in
            instances of hplc_data.
        calibrations : list of hplc_calibration instances
            Each element of the list must be a list itself. If more than one
            calibration is contained in an element, they must share the same
            time and spectral constraints, and in the advanced classical least
            squares algorithm, the respective calibration datasets are all used
            for spectral fitting.

        Returns
        -------
        None.

        """
        self.samples = samples
        self.calibrations = calibrations

    def simple_prediction(self, mode='cls'):
        """
        Predicts concentrations of n samples by the classical method.

        Based on first integrating elugrams at all wavelengths and subsequent
        either classical least squares fitting, principal component regression,
        or partial least squares regression of the resulting spectrum with the
        m calibration data present in self.calibrations. Thus, the separation
        of the spectral information is lost in the elugram regions present in
        the calibrations. Data are analyzed based on time limits and wavelength
        limits given in the calibrations. This procedure works well for
        baseline separated peaks and it is possible to use wavelength ranges
        (multivariate) or a single wavelength in case of classical least
        squares fitting (univariate, as routinely done in HPLC analysis).

        Parameters
        ----------
        mode : str, optional
            Calibration mode used for prediction. Allowed values are 'cls' for
            classical least squares, 'pcr' for principal component, and 'plsr'
            for partial least squares regression. The default is 'cls'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        predictions : ndarray
            Contains the predicted concentrations of the n samples for m
            components determined by the m calibrations by the method
            determined by mode. Shape is (n, m).

        """
        cals_squeezed = [
            item for sublist in self.calibrations for item in sublist]
        predictions = np.zeros((len(self.samples), len(cals_squeezed)))

        for sample_index, sample in enumerate(self.samples):
            for cal_index, calibration in enumerate(cals_squeezed):
                curr_data = sample.integrate_all_data(
                    time_limits=calibration.time_limits,
                    wavelength_limits=calibration.wavelength_limits
                    ).values
                if mode == 'cls':  # classical least squares
                    prediction = np.dot(
                        np.linalg.inv(np.dot(calibration.K.T, calibration.K)),
                        np.dot(
                            calibration.K.T,
                            curr_data[np.newaxis].T))
                elif mode == 'pcr':  # principal component regression
                    prediction = calibration.pcr_calibration.predict(
                        curr_data.T.reshape(1, -1),
                        calibration.pcr_components)
                elif mode == 'plsr':  # partial least squares regression
                    prediction = calibration.plsr_calibration.predict(
                        curr_data.T.reshape(1, -1),
                        calibration.plsr_components)
                else:
                    raise ValueError('No valid prediction mode given.')
                predictions[sample_index, cal_index] = prediction.item()

        return predictions

    def advanced_prediction(self, mode='cls'):  # mode not used at the moment
        """
        Predicts concentrations of n samples including chemometric methods.

        Based on first fitting all spectra from the different time points
        with the calibration data given with a classical least squares method.
        By giving more than one calibration for a certain time interval,
        overlapping peaks might be resolved based on differences in their
        spectra. Thus, the separation of the spectral information is used for
        data analysis. Data are analyzed based on time limits and wavelength
        limits given in the calibrations. This procedure works best if some
        separation of the different components in retention time and spectrum
        exists, more separation is better. Baseline separated peaks are however
        not necessary. It is only possible to use wavelength ranges
        (multivariate), using only a single wavelength (univariate) is not
        possible because the classical least squares fit of the spectra will
        fail.

        Raises
        ------
        ValueError
            In case calibration data is based on a single wavelength only.

        Returns
        -------
        predictions : ndarray
            Contains the predicted concentrations of the n samples for m
            components determined by the m calibrations. Shape is (n, m).

        """
        cal_set_sizes = [len(cal_set) for cal_set in self.calibrations]
        number_of_cals = sum(cal_set_sizes)
        predictions = np.zeros((len(self.samples), number_of_cals))

        index_counter = 0
        for curr_set_index, curr_cals in enumerate(self.calibrations):
            # calibrations in one set must have equal time and wavelength range
            Ks = []
            for calibration in curr_cals:
                Ks.append(np.squeeze(calibration.K))
            Ks = np.array(Ks).T
            if len(Ks.shape) == 1:
                raise ValueError(
                    'Singular wavelength used for multivariate calibration.')
            time_limits = curr_cals[0].time_limits
            wavelength_limits = curr_cals[0].wavelength_limits

            for sample_index, sample in enumerate(self.samples):
                sample_cropped = sample.crop_data(
                    time_limits=time_limits,
                    wavelength_limits=wavelength_limits)

                curr_pred = np.dot(
                    np.linalg.inv(np.dot(Ks.T, Ks)),
                    np.dot(Ks.T, sample_cropped.T)
                    )
                curr_pred = np.trapz(
                    np.squeeze(curr_pred), x=sample_cropped.index)

                predictions[
                    sample_index, index_counter:index_counter+cal_set_sizes[
                        curr_set_index]] = curr_pred
            index_counter = index_counter+cal_set_sizes[curr_set_index]

        return predictions


if __name__ == "__main__":

    def simulate_hplc_data(concentrations, retention_times,
                           spectrum_wavelengths, spectrum_amplitudes,
                           spectrum_widths,
                           wavelengths=np.linspace(200, 400, 201),
                           times=np.linspace(0, 10, 1001),
                           noise_level=0.05):
        """
        Simulate one measurement of HPLC 3D data.

        Absorption spectra are calculated as superpositions of Gaussian
        absorption bands and elugram peaks as individual Gaussian peaks (with
        currently fixed width, might be changed in the future). The number n of
        peaks and thus the number of components is not limited.

        Parameters
        ----------
        concentrations : list of floats
            List giving the concentrations of the n components present in the
            mixture. Length is n.
        retention_times : list of floats
            List giving the peak center retention times of the n components
            present in the mixture. Length is n..
        spectrum_wavelengths : list of lists
            Contains the absorption band center wavelengths for each spectrum
            of the n components as lists, so contains n lists. Each of the n
            lists may contain as many floats as necessary to describe the
            spectrum.
        spectrum_amplitudes : list of lists
            Contains the absorption band amplitudes for each spectrum
            of the n components as lists, so contains n lists in the same order
            and shape as in spectrum_wavelengths. Each of the n lists therefore
            contains as many items as in spectrum_wavelengths.
        spectrum_widths : list of lists
            Contains the absorption band widths as Gausiian sigma for each
            spectrum of the n components as lists, so contains n lists in the
            same order and shape as in spectrum_wavelengths. Each of the n
            lists therefore contains as many items as in spectrum_wavelengths.
        wavelengths : ndarray, optional
            Wavelengths used for calculating the absorption spectra. The
            default is np.linspace(200, 400, 201).
        times : ndarray, optional
            Time values used for calculation of the elugrams. The default is
            np.linspace(0, 10, 1001).
        noise_level : float, optional
            Data is superimposed with Gaussian noise if noise_level != 0. The
            default is 0.05. The bigger the value, the more noise is present.

        Returns
        -------
        data_3D : instance of hplc_data
            Contains the calculated data as an instance of hplc_data.

        """
        number_of_components = len(concentrations)
        # first step: pure component absorption spectra are calculated and
        # stored in uv_spectra
        uv_spectra = np.zeros((number_of_components, len(wavelengths)))
        for index, (curr_amp, curr_wl, curr_width) in enumerate(
                zip(spectrum_amplitudes, spectrum_wavelengths,
                    spectrum_widths)):
            curr_y_offset = len(curr_amp)*[0]
            curr_params = np.ravel(np.array(
                [curr_amp, curr_wl, curr_y_offset, curr_width]).T)

            uv_spectra[index] = calc_function(
                wavelengths, curr_params, 'Gauss')

        # second step: basic chromatogram shapes separately for each component
        # are calculated and stored in chromatograms
        chromatograms = np.zeros((number_of_components, len(times)))
        chrom_params = np.repeat([[1, 0, 0.2]], number_of_components, axis=0)
        chrom_params = np.insert(chrom_params, 1, retention_times, axis=1)
        for jj, curr_params in enumerate(chrom_params):
            chromatograms[jj] = calc_function(times, curr_params, 'Gauss')

        # third step: spectra and chromatrograms are combined to 3D dataset
        # and noise is added
        weighted_spectra = np.array(concentrations)[:, np.newaxis]*uv_spectra
        data_3D = np.dot(chromatograms.T, weighted_spectra)
        noise = np.random.standard_normal(data_3D.shape)*noise_level
        data_3D = hplc_data('DataFrame',
                            data=pd.DataFrame(data_3D + noise,
                                              index=times,
                                              columns=wavelengths))
        return data_3D

    # calculate simulated HPLC/DAD output for calibration
    calib_c = [0.1, 0.2, 0.3, 0.4, 0.5]
    calibration_1_data = []
    calibration_2_data = []
    calibration_3_data = []
    calibration_4_data = []
    for curr_c in calib_c:
        calibration_1_data.append(
            simulate_hplc_data([curr_c], [4.3], [[200, 250]], [[4, 0.8]],
                               [[15, 15]], noise_level=0.05)
            )
        calibration_2_data.append(
            simulate_hplc_data([curr_c], [5], [[200, 275]], [[4, 1.3]],
                               [[15, 12]], noise_level=0.05)
            )
        calibration_3_data.append(
            simulate_hplc_data([curr_c], [7.3], [[200, 275]], [[4, 1.3]],
                               [[15, 12]], noise_level=0.05)
            )
        calibration_4_data.append(
            simulate_hplc_data([curr_c], [7.3], [[200, 300]], [[4, 1.8]],
                               [[15, 12]], noise_level=0.05)
            )

    # generate hplc_calibration instance with simulated calibration data
    calibration_1 = hplc_calibration('hplc_data', calibration_1_data,
                                     calib_c, time_limits=[3, 6],
                                     wavelength_limits=[225, 300],
                                     plsr_components=2, pcr_components=2)

    calibration_1_uni = hplc_calibration('hplc_data', calibration_1_data,
                                         calib_c, time_limits=[3, 6],
                                         wavelength_limits=[250, 250])

    calibration_2 = hplc_calibration('hplc_data', calibration_2_data,
                                     calib_c, time_limits=[3, 6],
                                     wavelength_limits=[225, 300])

    calibration_3 = hplc_calibration('hplc_data', calibration_3_data,
                                     calib_c, time_limits=[6.1, 9],
                                     wavelength_limits=[225, 300],
                                     plsr_components=2, pcr_components=2)

    calibration_4 = hplc_calibration('hplc_data', calibration_4_data,
                                     calib_c, time_limits=[6.1, 9],
                                     wavelength_limits=[250, 350])

    # calculate samples with unknown concentration
    unknown_sample_1 = simulate_hplc_data(
        [0.35, 0.2], [4.3, 5], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]])

    unknown_sample_2 = simulate_hplc_data(
        [0.27, 0.55], [4.3, 5], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]])

    unknown_sample_3 = simulate_hplc_data(
        [3, 2], [4.3, 7.3], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]], noise_level=0.05)

    unknown_sample_4 = simulate_hplc_data(
        [0.9, 1.4, 0.7], [4.3, 5, 7.3], [[200, 250], [200, 275], [200, 300]],
        [[4, 0.8], [4, 1.3], [4, 1.8]], [[15, 15], [15, 12], [15, 12]])

    unknown_sample_5 = simulate_hplc_data(
        [0.49], [4.3], [[200, 250]],
        [[4, 0.8]], [[15, 15]])

    # predict unknown concentrations with multivariate calibrations
    predicted_concentrations = hplc_prediction(
        [unknown_sample_1, unknown_sample_2], [[calibration_1, calibration_2]])
    unknown_concentrations_simple = (
        predicted_concentrations.simple_prediction())
    unknown_concentrations_advanced = (
        predicted_concentrations.advanced_prediction())

    predicted_concentrations_2 = hplc_prediction(
        [unknown_sample_3], [[calibration_1], [calibration_3]])
    unknown_concentrations_simple_2 = (
        predicted_concentrations_2.simple_prediction())
    unknown_concentrations_advanced_2 = (
        predicted_concentrations_2.advanced_prediction())

    predicted_concentrations_3 = hplc_prediction(
        [unknown_sample_4, unknown_sample_5], [[calibration_1, calibration_2],
                                               [calibration_4]])
    unknown_concentrations_simple_3 = (
        predicted_concentrations_3.simple_prediction())
    unknown_concentrations_advanced_3 = (
        predicted_concentrations_3.advanced_prediction())

    print('Correct_concentrations:\n0.35, 0.2\n0.27, 0.55')
    print('Predicted concentrations (multivariate, simple):\n',
          unknown_concentrations_simple)
    print('Predicted concentrations (multivariate, advanced):\n',
          unknown_concentrations_advanced)

    print('Simple sample, simple (3, 2):\n', unknown_concentrations_simple_2)
    print('Simple sample, advanced (3, 2):\n',
          unknown_concentrations_advanced_2)
    print('Complex sample, simple (0.9, 1.4, 0.7):\n',
          unknown_concentrations_simple_3)
    print('Complex sample, advanced (0.9, 1.4, 0.7):\n',
          unknown_concentrations_advanced_3)

    # plot some data
    plt.figure(0)
    plt.plot(calibration_1.calibration_data[4].raw_data.columns,
             calibration_1.calibration_data[4].raw_data.loc[4.3, :],
             calibration_2.calibration_data[4].raw_data.columns,
             calibration_2.calibration_data[4].raw_data.loc[5, :])

    plt.figure(1)
    plt.plot(unknown_sample_1.raw_data.index,
             unknown_sample_1.raw_data.loc[:, 200])

    plt.figure()
    plt.plot(calibration_1.K)
    plt.plot(calibration_2.K)
