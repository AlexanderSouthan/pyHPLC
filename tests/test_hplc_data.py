# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:55:00 2021

@author: Alexander Southan
"""

import unittest

from src.pyHPLC.hplc_data import hplc_data


class TestHPLCData(unittest.TestCase):

    def test_hplc_data(self):

        test_data = hplc_data('import', file_name='tests/sample_data.txt')
        test_elugram = test_data.extract_elugram(250)
        cropped_data = test_data.crop_data(time_limits=[0, 10])
        test_spectrum = test_data.extract_spectrum(5)
