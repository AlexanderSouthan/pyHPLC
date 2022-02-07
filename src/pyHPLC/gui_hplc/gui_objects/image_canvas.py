# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:38:03 2019

@author: AlMaMi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class image_canvas(FigureCanvas):

    def __init__(self,parent=None,plot_title='Plot Title',scroll_type=None):
        self.plot_title = plot_title
        self.scroll_type = scroll_type

        self.fig,self.ax = plt.subplots(1,1)
        self.ax.set_title(self.plot_title)

        FigureCanvas.__init__(self,self.fig)
        self.parent = parent

        #FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        #FigureCanvas.updateGeometry(self)

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.plot_update()

    def set_image_data(self,data,vertical_axis=np.arange(0,10,1),horizontal_axis=np.arange(0,10,1)):
        self.data = data
        self.vertical_axis = vertical_axis
        self.horizontal_axis = horizontal_axis
        
        if self.scroll_type == 'z':
            self.slices = len(self.parent.parent.z_values_coded_processed)
            self.coordinates = self.parent.parent.z_values_processed
        elif self.scroll_type == 'y':
            self.slices = len(self.parent.parent.y_values_coded_processed)
            self.coordinates = self.parent.parent.y_values_processed
        elif self.scroll_type == 'x':
            self.slices = len(self.parent.parent.x_values_coded_processed)
            self.coordinates = self.parent.parent.x_values_processed
        else:
            self.slices = 1
        
        self.calc_curr_data(mode='initial')
            
        self.im = self.ax.imshow(self.curr_dataset,cmap='binary',extent=[self.horizontal_axis[0],self.horizontal_axis[-1],self.vertical_axis[0],self.vertical_axis[-1]])
        self.plot_update()
        
    def plot_update(self):
        self.calc_curr_data()
            
        self.im.set_data(self.curr_dataset)
        if self.scroll_type in ['x','y','z']:
            self.ax.set_title('slice %s/%s, %s = %s' % (self.ind+1,self.slices,self.scroll_type,self.coordinates[self.ind]))
        self.im.axes.figure.canvas.draw()
        
    def calc_curr_data(self,mode='continuous'):
        if mode=='initial': self.ind = self.slices//2
        
        if self.scroll_type == 'z':
            self.curr_dataset = self.parent.parent.raman_data.xy_slice(self.parent.parent.z_values_coded_processed[self.ind],col_index = self.parent.parent.projections_window.select_column_combo.currentText())
        elif self.scroll_type == 'y':
            self.curr_dataset = self.parent.parent.raman_data.xz_slice(self.parent.parent.y_values_coded_processed[self.ind],col_index = self.parent.parent.projections_window.select_column_combo.currentText())
        elif self.scroll_type == 'x':
            self.curr_dataset = self.parent.parent.raman_data.yz_slice(self.parent.parent.x_values_coded_processed[self.ind],col_index = self.parent.parent.projections_window.select_column_combo.currentText())
        else:
            self.curr_dataset = self.data