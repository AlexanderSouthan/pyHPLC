# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:40:26 2019

@author: AlMaMi
"""

import sys
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QComboBox, QWidget,
                             QLineEdit, QFileDialog, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QLabel, QAction, QDesktopWidget, 
                             QActionGroup, QMenu, QListWidget, QAbstractItemView)

from hplc_import_window import hplc_import_window
from hplc_calibration_window import hplc_calibration_window
from hplc_visualization_window import hplc_visualization_window
from pyAnalytics.hplc_prediction import hplc_prediction


class main_window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

        self.hplc_datasets = {}
        self.hplc_file_names = {}
        self.hplc_calibrations = {}

    def init_window(self):
        self.setGeometry(500, 500, 1200, 100)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('HPLC data analysis')

        self.statusBar().showMessage('Welcome')
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        visualize_menu = menubar.addMenu('&Visualize')
        preprocess_menu = menubar.addMenu('&Preprocessing')
        calibration_menu = menubar.addMenu('&Calibration')
        analysis_menu = menubar.addMenu('&Analyze data')

        import_filter_menu = QMenu('Dataset import filter', self)
        self.import_filter_group = QActionGroup(import_filter_menu)
        # Any entry in the following list generates a new entry in import
        # filter submenu
        self.import_filters = ['3D ASCII data', 'Pickled hplc_data object']
        for import_filter in self.import_filters:
            import_action = QAction(import_filter, import_filter_menu,
                                    checkable=True,
                                    checked=import_filter==self.import_filters[0])
            import_filter_menu.addAction(import_action)
            self.import_filter_group.addAction(import_action)
        self.import_filter_group.setExclusive(True)
        file_menu.addMenu(import_filter_menu)

        import_dataset_action = QAction('Import dataset', self)
        import_dataset_action.setShortcut('Ctrl+I')
        import_dataset_action.setStatusTip('Import dataset')
        import_dataset_action.triggered.connect(self.open_import_window)

        open_dataset_action = QAction('Open dataset from file', self)
        open_dataset_action.setShortcut('Ctrl+O')
        open_dataset_action.setStatusTip('Open dataset from file')
        open_dataset_action.triggered.connect(self.open_dataset)

        save_dataset_action = QAction('Save dataset to file', self)
        save_dataset_action.setShortcut('Ctrl+S')
        save_dataset_action.setStatusTip('Save dataset to file')
        save_dataset_action.triggered.connect(self.save_dataset)

        open_calibration_action = QAction('Open calibration from file', self)
        open_calibration_action.setStatusTip('Open calibration from file')
        open_calibration_action.triggered.connect(self.open_calibration)

        save_calibration_action = QAction('Save calibration to file', self)
        save_calibration_action.setStatusTip('Save calibration to file')
        save_calibration_action.triggered.connect(self.save_calibration)

        file_menu.addMenu(import_filter_menu)
        file_menu.addAction(import_dataset_action)
        file_menu.addAction(open_dataset_action)
        file_menu.addAction(save_dataset_action)
        file_menu.addAction(open_calibration_action)
        file_menu.addAction(save_calibration_action)

        elugram_viewer_action = QAction('2D elugram viewer', self)
        elugram_viewer_action.triggered.connect(self.open_elugram_window)

        spectrum_viewer_action = QAction('2D spectrum viewer', self)
        spectrum_viewer_action.triggered.connect(self.open_spectrum_window)

        visualize_menu.addAction(elugram_viewer_action)
        visualize_menu.addAction(spectrum_viewer_action)

        calibration_viewer_action = QAction('Calibration wizard', self)
        calibration_viewer_action.triggered.connect(
            self.open_calibration_window)

        calibration_menu.addAction(calibration_viewer_action)

        simple_cls_action = QAction('Simple cls analysis', self)
        simple_cls_action.triggered.connect(
            self.analyze_dataset)
        
        analysis_menu.addAction(simple_cls_action)

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        # Interesting preprocessing options: Baseline correction, normalize
        # with internal standard. Should both be integrated into hplc_data.
        self.dataset_selection_label = QLabel('<b>Active dataset</b>')
        self.dataset_selection_combo = QComboBox()

        self.calibration_selection_label = QLabel('<b>Active calibration</b>')
        self.calibration_selection_list = QListWidget()
        self.calibration_selection_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def position_widgets(self):
        self.dataset_selection_layout = QVBoxLayout()
        self.dataset_selection_layout.addWidget(self.dataset_selection_label)
        self.dataset_selection_layout.addWidget(self.dataset_selection_combo)
        self.dataset_selection_layout.addStretch(1)

        self.calibration_selection_layout = QVBoxLayout()
        self.calibration_selection_layout.addWidget(
            self.calibration_selection_label)
        self.calibration_selection_layout.addWidget(
            self.calibration_selection_list)

        self.hplc_data_selection_layout = QHBoxLayout()
        self.hplc_data_selection_layout.addLayout(
            self.dataset_selection_layout)
        self.hplc_data_selection_layout.addLayout(
            self.calibration_selection_layout)
        self.hplc_data_selection_layout.addStretch(1)

        self.grid_container.addLayout(
            self.hplc_data_selection_layout, 0, 1, 1, 1)

        # self.grid_container.addWidget(self.import_options_label, *(1, 1), 1, 1)
        # self.spectra_plot_limits_layout.addStretch(1)

    def connect_event_handlers(self):
        self.dataset_selection_combo.currentIndexChanged.connect(
            self.update_windows)
        self.calibration_selection_list.itemClicked.connect(self.set_active_calibrations)

    def open_calibration_window(self):
        self.hplc_calibration_window = hplc_calibration_window(self)
        self.hplc_calibration_window.show()

    def open_elugram_window(self):
        self.hplc_elugram_window = hplc_visualization_window(self, mode='elugram')
        self.hplc_elugram_window.show()

    def open_spectrum_window(self):
        self.hplc_spectrum_window = hplc_visualization_window(self, mode='spectrum')
        self.hplc_spectrum_window.show()

    def open_import_window(self):
        self.hplc_import_window = hplc_import_window(self)
        self.hplc_import_window.show()

    def analyze_dataset(self):
        # Analysis is currently performed on one elugram region only.
        # hplc_prediction can process multiple regions simultaneously, so that
        # still needs to be used here.
        curr_dataset = self.hplc_datasets[self.dataset_selection_combo.currentText()]
        curr_calibrations = []
        for calib in self.active_calibrations:
            curr_calibrations.append(self.hplc_calibrations[calib])
        # curr_calibration = self.hplc_calibrations[self.calibration_selection_combo.currentText()]

        predicted_concentrations = hplc_prediction(
            curr_dataset, [curr_calibrations])

        print('Simple:', predicted_concentrations.simple_prediction())
        print('Advanced:', predicted_concentrations.advanced_prediction())

    def update_windows(self):
        try:
            self.hplc_elugram_window.set_active_dataset()
        except:
            pass
        try:
            self.hplc_calibration_window.set_active_dataset()
        except:
            pass

    def open_dataset(self):
        file_type = 'HPLC dataset file (*.hplc)'
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open HPLC dataset file', filter=file_type)
        dataset_name = file_name.split('/')[-1]

        if file_name != '':
            with open(file_name, 'rb') as filehandle:
                self.hplc_datasets[dataset_name] = pickle.load(filehandle)

            datasets_import_paths = []
            for curr_dataset in self.hplc_datasets[dataset_name]:
                datasets_import_paths.append(curr_dataset.import_path)
            self.hplc_file_names[dataset_name] = datasets_import_paths

            dataset_names = [
                self.dataset_selection_combo.itemText(i)
                for i in range(self.dataset_selection_combo.count())]

            if dataset_name not in dataset_names:
                self.dataset_selection_combo.addItem(dataset_name)

    def save_dataset(self):
        curr_dataset = self.dataset_selection_combo.currentText()

        file_type = 'HPLC dataset file (*.hplc)'
        file_name, _ = QFileDialog.getSaveFileName(
            self, 'Save active HPLC dataset file', curr_dataset + '.hplc',
            filter=file_type)

        if file_name != '':
            with open(file_name, 'wb') as filehandle:
                pickle.dump(self.hplc_datasets[curr_dataset], filehandle)

    def open_calibration(self):
        file_type = 'HPLC calibration file (*.calib)'
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open HPLC calibration file', filter=file_type)
        calibration_name = file_name.split('/')[-1]

        if file_name != '':
            if calibration_name not in self.hplc_calibrations.keys():
                self.calibration_selection_list.addItem(calibration_name)

            with open(file_name, 'rb') as filehandle:
                self.hplc_calibrations[calibration_name] = pickle.load(
                    filehandle)

            # calibration_names = [
            #     self.calibration_selection_combo.itemText(i)
            #     for i in range(self.calibration_selection_combo.count())]

            # if calibration_name not in calibration_names:
            #     self.calibration_selection_combo.addItem(calibration_name)

            # # the rest is for the QListWidget, currently experimental
            # calibration_names_1 = []
            # counter = 0
            # finished = False
            # while finished == False:
            #     curr_item = self.calibration_selection_list.item(counter)
            #     if curr_item is None:
            #         finished = True
            #     else:
            #         calibration_names_1.append(curr_item.text())
            #         counter += 1

    def save_calibration(self):
        # curr_calibration = self.calibration_selection_combo.currentText()

        for curr_calibration in self.active_calibrations:
            file_type = 'HPLC calibration file (*.calib)'
            file_name, _ = QFileDialog.getSaveFileName(
                self, 'Save active HPLC calibration file',
                curr_calibration + '.calib', filter=file_type)

            if file_name != '':
                with open(file_name, 'wb') as filehandle:
                    pickle.dump(
                        self.hplc_calibrations[curr_calibration], filehandle)

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_active_calibrations(self):
        selected = self.calibration_selection_list.selectedItems()
        self.active_calibrations = []
        for curr_item in selected:
            self.active_calibrations.append(curr_item.text())

app = QApplication(sys.argv)
   
window = main_window()

window.show()
#app.exec_()
sys.exit(app.exec_())
   


