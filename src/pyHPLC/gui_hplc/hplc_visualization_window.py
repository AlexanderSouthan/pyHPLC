# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:33:44 2020

@author: aso
"""
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget,QComboBox,QCheckBox,QSpinBox,QApplication,
                             QMainWindow,QSizePolicy,QLineEdit,QGridLayout,
                             QHBoxLayout,QLabel,QDesktopWidget)

# import preprocessing.baseline_correction
# import data_objects.hplc_data
from gui_objects.plot_canvas import plot_canvas


class hplc_visualization_window(QMainWindow):
    def __init__(self, parent, mode='elugram'):
        super().__init__()
        self.parent = parent
        self.mode = mode

        self.init_window()
        self.define_widgets()

        self.set_active_dataset()

        self.position_widgets()
        self.connect_event_handlers()
        self.update_elugram_plot()

    def init_window(self):
        self.setGeometry(500, 500, 1200, 900)  # xPos, yPos, width, heigth
        self.center()  # c enter function is defined below
        self.setWindowTitle('2D {} viewer'.format(self.mode))

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.select_sample_label = QLabel('<b>Sample number</b>')
        self.select_sample_spinbox = QSpinBox()
        # self.select_sample_spinbox.setRange(
        #     1, len(self.parent.hplc_datasets[self.active_dataset]))

        self.sample_name_label = QLabel('<b>Sample name</b>')
        self.sample_name_lineedit = QLineEdit()
        self.sample_name_lineedit.setReadOnly(True)

        self.show_all_label = QLabel('Show all {}s'.format(self.mode))
        self.show_all_checkbox = QCheckBox()
        self.show_all_checkbox.setChecked(False)

        self.elugram_plot = plot_canvas(
            plot_title='{} plot'.format(self.mode),
            x_axis_title='elution time [min]' if self.mode=='elugram'
            else 'wavelength [nm]')

        self.data_selection_combo = QComboBox()
        self.data_selection_combo.addItems(
            ['{}s'.format(self.mode), '{}s + projections'.format(self.mode)])

        self.wl_selection_label = QLabel('Wavelength' if self.mode=='elugram'
                                         else 'Elution time')
        self.wl_selection_combo = QComboBox()

        self.xlimits_label = QLabel('x limits')
        self.lower_x_combo = QComboBox()
        self.upper_x_combo = QComboBox()

    def position_widgets(self):
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(self.select_sample_label)
        self.header_layout.addWidget(self.select_sample_spinbox)
        self.header_layout.addWidget(self.sample_name_label)
        self.header_layout.addWidget(self.sample_name_lineedit)
        self.header_layout.addWidget(self.show_all_label)
        self.header_layout.addWidget(self.show_all_checkbox)
        self.header_layout.addWidget(self.data_selection_combo)

        self.header2_layout = QHBoxLayout()
        self.header2_layout.addWidget(self.wl_selection_label)
        self.header2_layout.addWidget(self.wl_selection_combo)
        self.header2_layout.addWidget(self.xlimits_label)
        self.header2_layout.addWidget(self.lower_x_combo)
        self.header2_layout.addWidget(self.upper_x_combo)
        self.header2_layout.addStretch(1)

        self.grid_container.addLayout(self.header_layout, *(0, 0), 1, 1)
        self.grid_container.addLayout(self.header2_layout, *(1, 0), 1, 1)
        self.grid_container.addWidget(self.elugram_plot, *(2, 0), 1, 1)

    def connect_event_handlers(self):
        self.select_sample_spinbox.valueChanged.connect(
            self.update_elugram_plot)
        self.show_all_checkbox.stateChanged.connect(self.update_elugram_plot)
        self.data_selection_combo.currentIndexChanged.connect(
            self.update_elugram_plot)
        self.wl_selection_combo.currentIndexChanged.connect(
            self.update_elugram_plot)
        self.lower_x_combo.currentIndexChanged.connect(
            self.update_elugram_plot)
        self.upper_x_combo.currentIndexChanged.connect(
            self.update_elugram_plot)

    def update_elugram_plot(self):
        if self.show_all_checkbox.checkState() == 2:
            mode = 'all'
            self.select_sample_spinbox.setEnabled(False)
        else:
            mode = 'single'
            self.select_sample_spinbox.setEnabled(True)

        current_value = float(self.wl_selection_combo.currentText())

        current_sample_number = self.select_sample_spinbox.value() - 1
        self.sample_name_lineedit.setText(
            self.parent.hplc_datasets[
                self.active_dataset][current_sample_number].import_path)

        if self.mode == 'elugram':
            dim = 'time'
        elif self.mode == 'spectrum':
            dim = 'wavelength'

        self.elugram_plot.axes.clear()
        x_limits = np.sort(
            [float(self.lower_x_combo.currentText()),
             float(self.upper_x_combo.currentText())])

        if mode == 'single':
            plot_items = [current_sample_number]
        elif mode == 'all':
            plot_items = np.arange(
                len(self.parent.hplc_file_names[self.active_dataset]))

        curr_y = pd.DataFrame()
        y_min = 0
        y_max = 0
        for ii in plot_items:
            colors = ['k']
            if self.mode == 'elugram':
                curr_values = self.parent.hplc_datasets[
                    self.active_dataset][ii].extract_elugram(
                        current_value, time_limits=x_limits)
            elif self.mode == 'spectrum':
                curr_values = self.parent.hplc_datasets[
                    self.active_dataset][ii].extract_spectrum(
                        current_value, wavelength_limits=x_limits)
            curr_y = pd.concat([curr_y, curr_values], axis=1)
            curr_x = curr_y.index.to_numpy()

            if self.data_selection_combo.currentText() in [
                    '{}s + projections'.format(self.mode)]:
                curr_y = pd.concat(
                    [curr_y, self.parent.hplc_datasets[
                        self.active_dataset][ii].generate_projections(
                            dim=dim,
                            time_limits=x_limits
                            if self.mode=='elugram' else None,
                            wavelength_limits=x_limits
                            if self.mode=='spectrum' else None)
                        ],
                    axis=1)
                colors.extend(['r', 'b', 'g'])

            y_min = min(np.min(curr_y.values), y_min)
            y_max = max(np.max(curr_y.values), y_max)

            for curr_line, curr_color in zip(curr_y.columns, colors):
                self.elugram_plot.plot(curr_x, curr_y[curr_line],
                                       pen=curr_color)

        self.elugram_plot.axes.set_xlim(left=x_limits[0], right=x_limits[1])
        self.elugram_plot.axes.set_ylim(
            bottom=y_min, top=y_max if y_max!=y_min else y_max+1)

        #self.elugram_plot.axes.axvline(x=INTEGRATION_LIMIT_LEFT,color='k',dashes=(5,5))
        #self.elugram_plot.axes.axvline(x=INTEGRATION_LIMIT_RIGHT,color='k',dashes=(5,5))

        self.elugram_plot.draw()

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_active_dataset(self):
        self.active_dataset = self.parent.dataset_selection_combo.currentText()

        self.select_sample_spinbox.setRange(
            1, len(self.parent.hplc_datasets[self.active_dataset]))
        self.sample_name_lineedit.setText(
            self.parent.hplc_datasets[self.active_dataset][0].import_path)

        self.wl_selection_combo.blockSignals(True)
        self.lower_x_combo.blockSignals(True)
        self.upper_x_combo.blockSignals(True)
        self.wl_selection_combo.clear()
        self.lower_x_combo.clear()
        self.upper_x_combo.clear()

        wl_items = [
            '{:.2f}'.format(x) for x in
            self.parent.hplc_datasets[self.active_dataset][0].wavelengths]
        time_items = [
            '{:.2f}'.format(x) for x in
            self.parent.hplc_datasets[
                self.active_dataset][0].raw_data.index.to_numpy()]

        if self.mode == 'elugram':
            x_dimension = time_items
            lost_dimension = wl_items
        elif self.mode == 'spectrum':
            x_dimension = wl_items
            lost_dimension = time_items

        self.wl_selection_combo.addItems(lost_dimension)
        self.lower_x_combo.addItems(x_dimension)
        self.upper_x_combo.addItems(x_dimension)

        self.upper_x_combo.setCurrentIndex(len(x_dimension)-1)

        self.wl_selection_combo.blockSignals(False)
        self.lower_x_combo.blockSignals(False)
        self.upper_x_combo.blockSignals(False)
        
        self.update_elugram_plot()