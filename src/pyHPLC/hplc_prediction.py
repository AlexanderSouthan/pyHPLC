# -*- coding: utf-8 -*-
"""
Contains class hplc_prediction.

Can only be used in combination with hplc_data and hplc_calibration.
"""

import numpy as np

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
                        np.dot(calibration.K.T, curr_data[np.newaxis].T))
                elif mode == 'pcr':  # principal component regression
                    prediction = calibration.pcr_calibration.predict(
                        curr_data.T.reshape(1, -1),
                        calibration.pcr_components).values
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
                    wavelength_limits=wavelength_limits, to_processed=False)

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
