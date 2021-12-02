# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:41:27 2019

@author: AlMaMi
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy

class plot_canvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100,plot_title='Plot Title',x_axis_title='x values'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.plot_title = plot_title
        self.x_label = x_axis_title

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot(self,x_Values,y_Values,pen='y',mode='line',error_data=None):
        if len(x_Values) > 1:
            format_string = '-'
        else:
            format_string = '+'
            
        if mode == 'line':
            self.axes.plot(x_Values,y_Values,format_string,color=pen)
        elif mode == 'line_symbol':
            self.axes.plot(
                x_Values, y_Values, format_string, color=pen, marker='o')
        elif mode == 'error_bar':
            self.axes.errorbar(x_Values,y_Values,fmt=format_string,yerr=error_data,color=pen,capsize=1,elinewidth=0,ecolor='r')
        elif mode == 'scatter':
            self.axes.scatter(x_Values,y_Values,color=pen)
        self.axes.set_title(self.plot_title)
        self.axes.set_xlabel(self.x_label)
        self.draw()