# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:44:12 2019

@author: AlMaMi
"""
from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget, QGridLayout,
                             QDesktopWidget, QLabel, QPushButton, QTextEdit,
                             QHBoxLayout, QVBoxLayout, QFileDialog, QLineEdit)

from gui_objects.plot_canvas import plot_canvas
from pyAnalytics.hplc_calibration import hplc_calibration
# from gui_objects.image_canvas import image_canvas
# from pyAnalytics.hplc_data import hplc_data


class hplc_calibration_window(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.set_active_dataset()

        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

        self.update_calibration_parameters()

        # self.init_combos()

    def init_window(self):
        self.setGeometry(500,500,1200,900) #xPos,yPos,width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('HPLC calibration wizard')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)
        
    def define_widgets(self):
        self.wl_limits_label = QLabel('Wavelength limits')
        self.time_limits_label = QLabel('Time limits')
        self.calibration_concentrations_label = QLabel(
            'Calibration concentrations')
        self.wl_limits_low_lineedit = QLineEdit('')
        self.wl_limits_high_lineedit = QLineEdit('')
        self.time_limits_low_lineedit = QLineEdit('')
        self.time_limits_high_lineedit = QLineEdit('')
        self.calibration_concentrations_lineedit = QLineEdit('')
        self.update_calibration_button = QPushButton('Preview calibration')

        self.calibration_plot = plot_canvas(
            plot_title='Calibration result',x_axis_title = 'concentration')
        
        self.calibration_name_label = QLabel('Calibration name')
        self.calibration_name_lineedit = QLineEdit('calibration_1')
        self.save_calibration_button = QPushButton('Accept calibration')

    def position_widgets(self):

        self.calibration_limits_label_layout = QVBoxLayout()
        self.calibration_limits_label_layout.addWidget(self.wl_limits_label)
        self.calibration_limits_label_layout.addWidget(self.time_limits_label)
        self.calibration_limits_label_layout.addWidget(
            self.calibration_concentrations_label)

        self.wl_limits_lineedit_layout = QHBoxLayout()
        self.wl_limits_lineedit_layout.addWidget(self.wl_limits_low_lineedit)
        self.wl_limits_lineedit_layout.addWidget(self.wl_limits_high_lineedit)

        self.time_limits_lineedit_layout = QHBoxLayout()
        self.time_limits_lineedit_layout.addWidget(
            self.time_limits_low_lineedit)
        self.time_limits_lineedit_layout.addWidget(
            self.time_limits_high_lineedit)

        self.calibration_limits_lineedit_layout = QVBoxLayout()
        self.calibration_limits_lineedit_layout.addLayout(
            self.wl_limits_lineedit_layout)
        self.calibration_limits_lineedit_layout.addLayout(
            self.time_limits_lineedit_layout)
        self.calibration_limits_lineedit_layout.addWidget(
            self.calibration_concentrations_lineedit)

        self.calibration_limits_layout = QHBoxLayout()
        self.calibration_limits_layout.addLayout(
            self.calibration_limits_label_layout)
        self.calibration_limits_layout.addLayout(
            self.calibration_limits_lineedit_layout)

        self.save_calibration_layout = QHBoxLayout()
        self.save_calibration_layout.addWidget(self.calibration_name_label)
        self.save_calibration_layout.addWidget(self.calibration_name_lineedit)
        self.save_calibration_layout.addWidget(self.save_calibration_button)

        self.calibration_layout = QVBoxLayout()
        self.calibration_layout.addLayout(self.calibration_limits_layout)
        self.calibration_layout.addWidget(self.update_calibration_button)
        self.calibration_layout.addWidget(self.calibration_plot)
        self.calibration_layout.addLayout(self.save_calibration_layout)

        # self.calibration_layout.addStretch(1)

        self.grid_container.addLayout(self.calibration_layout, 0, 1, 1, 1)

    def connect_event_handlers(self):
        self.wl_limits_low_lineedit.editingFinished.connect(
            self.update_calibration_parameters)
        self.wl_limits_high_lineedit.editingFinished.connect(
            self.update_calibration_parameters)
        self.time_limits_low_lineedit.editingFinished.connect(
            self.update_calibration_parameters)
        self.time_limits_high_lineedit.editingFinished.connect(
            self.update_calibration_parameters)
        self.update_calibration_button.clicked.connect(self.update_calibration)
        self.save_calibration_button.clicked.connect(self.save_calibration)
        # self.calibration_data_textedit.textChanged.connect(self.update_processing_parameters)

    def update_calibration_parameters(self):
        lower_wl_limit = self.wl_limits_low_lineedit.text()
        lower_wl_limit = float(lower_wl_limit) if lower_wl_limit != '' else None
        upper_wl_limit = self.wl_limits_high_lineedit.text()
        upper_wl_limit = float(upper_wl_limit) if upper_wl_limit != '' else None

        lower_time_limit = self.time_limits_low_lineedit.text()
        lower_time_limit = float(lower_time_limit) if lower_time_limit != '' else None
        upper_time_limit = self.time_limits_high_lineedit.text()
        upper_time_limit = float(upper_time_limit) if upper_time_limit != '' else None

        calibration_concentrations = (
            self.calibration_concentrations_lineedit.text().split(',')
            if self.calibration_concentrations_lineedit.text() != '' else [])
        calibration_concentrations = [
            float(i) for i in calibration_concentrations]

        self.calibration_parameters = {
            'wl_limits': [lower_wl_limit, upper_wl_limit],
            'time_limits': [lower_time_limit, upper_time_limit],
            'concentrations': calibration_concentrations}

    def update_calibration(self):
        assert len(self.calibration_parameters['concentrations']) == len(self.parent.hplc_datasets[self.active_dataset]), 'Number of concentrations is not equal to number of calibration samples.'
        self.calibration = hplc_calibration(
            'hplc_data', self.parent.hplc_datasets[self.active_dataset],
            self.calibration_parameters['concentrations'],
            time_limits=self.calibration_parameters['time_limits'],
            wavelength_limits=self.calibration_parameters['wl_limits'])
        
        self.calibration_plot.axes.clear()
        self.calibration_plot.plot(
            self.calibration_parameters['concentrations'],
            self.calibration.calibration_integrated.T, mode='line_symbol')

    def save_calibration(self):
        calibration_name = self.calibration_name_lineedit.text()

        if calibration_name not in self.parent.hplc_calibrations.keys():
            self.parent.calibration_selection_list.addItem(calibration_name)

        self.parent.hplc_calibrations[calibration_name] = []

        self.parent.hplc_calibrations[calibration_name] = (
            self.calibration)

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_active_dataset(self):
        self.active_dataset = self.parent.dataset_selection_combo.currentText()

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())