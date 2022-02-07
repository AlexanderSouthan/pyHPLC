# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:54:01 2022

@author: aso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyDataFitting.nonlinear_regression import calc_function
from pyHPLC import hplc_data
from pyHPLC import hplc_calibration
from pyHPLC import hplc_prediction

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

############ Step 1 ###################
# Calculate simulated HPLC/DAD output for calibration. Four different
# components are simulated, each having a very distinct peak in the elugram
# and a unique UV/Vis absorption spectrum.
calib_c = [0.1, 0.2, 0.3, 0.4, 0.5]

retention_times = [4.3, 5, 7.3, 7.3]
lambda_max = [[200, 250], [200, 275], [200, 275], [200, 300]]
amps = [[4, 0.8], [4, 1.3], [4, 1.3], [4, 1.8]]
widths = [[15, 15], [15, 12], [15, 12], [15, 12]]
noise_levels = [0.05, 0.05, 0.05, 0.05]
calibration_data = [[], [], [], []]
for curr_c in calib_c:
    for idx, _ in enumerate(calibration_data):
        calibration_data[idx].append(
            simulate_hplc_data(
                [curr_c], [retention_times[idx]], [lambda_max[idx]],
                [amps[idx]], [widths[idx]], noise_level=noise_levels[idx]))


# generate hplc_calibration instance with simulated calibration data
calibrations = []
time_limits = [[3, 6], [3, 6], [6.1, 9], [6.1, 9]]
wl_limits = [[225, 300], [225, 300], [250, 350], [250, 350]]
for curr_cal_data, curr_time, curr_wl in zip(calibration_data, time_limits, wl_limits):
    calibrations.append(hplc_calibration('hplc_data', curr_cal_data,
                                         calib_c, time_limits=curr_time,
                                         wavelength_limits=curr_wl,
                                         plsr_components=4, pcr_components=4))

calibration_1_uni = hplc_calibration('hplc_data', calibration_data[0],
                                     calib_c, time_limits=[3, 6],
                                     wavelength_limits=[250, 250])

# Plot elugrams of pure components. Components 3 and 4 have identical retention
# times, but different sprectra (see below).
fig1, ax1 = plt.subplots()
for idx, curr_cal in enumerate(calibrations):
    ax1.plot(curr_cal.calibration_data[4].extract_elugram(200), label='Component {}'.format(idx+1))
ax1.legend()
ax1.set_xlabel('Retention time [min]')
ax1.set_ylabel('Absorption [a.u.]')
ax1.set_title('Elugrams of the four pure components.')

# Plot spectra of pure components.
fig2, ax2 = plt.subplots()
for idx, curr_cal in enumerate(calibrations):
    ax2.plot(curr_cal.calibration_data[4].extract_spectrum(retention_times[idx]), label='Component {}'.format(idx+1))
ax2.legend()
ax2.set_xlabel('Wavelength [nm]')
ax2.set_ylabel('Absorption [a.u.]')
ax2.set_title('Absorption spectra of the four pure components at the peak max.')



############ Step 2 ###################
# Calculate samples as mixtures of the four components above. Each entry in the
# following list is one mixture.
mix_conc = [[0.35, 0.2, 0, 0],
            [0.27, 0.55, 0, 0],
            [3, 0, 2, 0],
            [0.9, 1.4, 0, 0.7],
            [0.49, 0, 0, 0]]

unknown_samples = []
for curr_mix_c in mix_conc:
    unknown_samples.append(simulate_hplc_data(
        curr_mix_c, retention_times, lambda_max,
        amps, widths))

# Plot the elugrams of the mixtures
fig3, ax3 = plt.subplots()
for ii, curr_mix in enumerate(unknown_samples):
    ax3.plot(curr_mix.extract_elugram(210), label='Mixture {}'.format(ii+1))
ax3.legend()
ax3.set_xlabel('Retention time [min]')
ax3.set_ylabel('Absorption [a.u.]')
ax3.set_title('Elugrams of the five mixtures.')



############ Step 3 ###################
real_concentrations = pd.DataFrame(
    mix_conc, index=['mix_1', 'mix_2', 'mix_3', 'mix_4', 'mix_5']).T

# predict unknown concentrations with multivariate calibrations
predicted_concentrations = hplc_prediction(
    unknown_samples, [calibrations[0:2], calibrations[2:4]])
unknown_concentrations_cls = pd.DataFrame(
    predicted_concentrations.simple_prediction(mode='cls'),
    index=['mix_1_pred', 'mix_2_pred', 'mix_3_pred', 'mix_4_pred', 'mix_5_pred']).T
unknown_concentrations_pcr = pd.DataFrame(
    predicted_concentrations.simple_prediction(mode='pcr'),
    index=['mix_1_pred', 'mix_2_pred', 'mix_3_pred', 'mix_4_pred', 'mix_5_pred']).T
unknown_concentrations_advanced = pd.DataFrame(
    predicted_concentrations.advanced_prediction(),
    index=['mix_1_pred', 'mix_2_pred', 'mix_3_pred', 'mix_4_pred', 'mix_5_pred']).T


print('Correct_concentrations:\n', real_concentrations)
print('Predicted concentrations (multivariate, simple, CLS):\n',
      unknown_concentrations_cls)
print('Predicted concentrations (multivariate, simple, PCR):\n',
      unknown_concentrations_pcr)
print('Predicted concentrations (multivariate, advanced):\n',
      unknown_concentrations_advanced)


# # plot some data
# fig1, ax1 = plt.subplots()
# ax1.plot(calibration_1.calibration_data[4].extract_spectrum(4.3))
# ax1.plot(calibration_2.calibration_data[4].extract_spectrum(5))

# fig2, ax2 = plt.subplots()
# ax2.plot(unknown_sample_1.raw_data.index,
#          unknown_sample_1.raw_data.loc[:, 200])

# fig3, ax3 = plt.subplots()
# ax3.plot(calibration_1.K)
# ax3.plot(calibration_2.K)
