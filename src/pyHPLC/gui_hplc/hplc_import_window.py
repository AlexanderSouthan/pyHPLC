# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:40:26 2019

@author: AlMaMi
"""


import numpy as np
import pandas as pd
import sys
import copy
import pickle
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QComboBox, QWidget,
                             QLineEdit, QFileDialog, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QTextEdit, QLabel, QToolTip, QAction,
                             qApp, QPushButton, QDesktopWidget, QCheckBox)

# import own modules ################
#from gui_objects.plot_canvas import plot_canvas
from pyAnalytics.hplc_data import hplc_data
from hplc_calibration_window import hplc_calibration_window
from hplc_visualization_window import hplc_visualization_window
#####################################


class hplc_import_window(QMainWindow):

    def __init__(self, parent):
        super().__init__()  # constructor of for parent object class is called
        self.parent = parent
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):

        self.setGeometry(500, 500, 600, 600)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('HPLC data import wizard')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.add_import_data_button = QPushButton('Add data files for import')
        self.clear_import_data_button = QPushButton('Clear')
        self.import_data_textedit = QTextEdit(readOnly=True)
        self.import_dataset_name_label = QLabel('<b>Dataset name</b>')
        self.import_dataset_name_lineedit = QLineEdit('import')
        self.import_data_button = QPushButton('Import data')

    def position_widgets(self):
        self.import_dataset_name_layout = QHBoxLayout()
        self.import_dataset_name_layout.addWidget(
            self.import_dataset_name_label)
        self.import_dataset_name_layout.addWidget(
            self.import_dataset_name_lineedit)

        self.import_data_layout = QVBoxLayout()
        self.import_data_layout.addWidget(self.add_import_data_button)
        self.import_data_layout.addWidget(self.clear_import_data_button)
        self.import_data_layout.addWidget(self.import_data_textedit)
        self.import_data_layout.addLayout(self.import_dataset_name_layout)
        self.import_data_layout.addWidget(self.import_data_button)

        self.grid_container.addLayout(self.import_data_layout, 0, 1, 1, 1)

    def connect_event_handlers(self):
        self.add_import_data_button.clicked.connect(self.add_import_data_path)
        self.clear_import_data_button.clicked.connect(self.import_data_textedit.clear)
        self.import_data_button.clicked.connect(self.import_data)

    def add_import_data_path(self):
        import_files = QFileDialog.getOpenFileNames(
            self, "Select data files for import")[0]
        for curr_file in import_files:
            self.import_data_textedit.append(curr_file)

    def import_data(self):
        import_dataset_name = self.import_dataset_name_lineedit.text()
        self.parent.hplc_datasets[import_dataset_name] = []
        self.parent.hplc_file_names[import_dataset_name] = (
            self.import_data_textedit.toPlainText().split('\n'))

        import_mode = self.parent.import_filter_group.checkedAction().text()

        for curr_file in tqdm(self.parent.hplc_file_names[import_dataset_name]):
            if import_mode == self.parent.import_filters[0]:  # 3D ASCII data
                self.parent.hplc_datasets[import_dataset_name].append(
                    hplc_data('import', full_path=curr_file))
            elif import_mode == self.parent.import_filters[1]:  # Pickled hplc_data object
                with open(curr_file, 'rb') as filehandle:
                    imported_data = pickle.load(filehandle)
                self.parent.hplc_datasets[import_dataset_name].append(
                    imported_data)
                self.parent.hplc_datasets[import_dataset_name][-1].import_path = curr_file

        dataset_names = [
            self.parent.dataset_selection_combo.itemText(i)
            for i in range(self.parent.dataset_selection_combo.count())]
        if import_dataset_name not in dataset_names:
            self.parent.dataset_selection_combo.addItem(import_dataset_name)

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
