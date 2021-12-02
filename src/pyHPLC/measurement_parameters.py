# -*- coding: utf-8 -*-
"""
Container for parameters of measurements which should be documented.

Accepts a Pandas DataFrame as input. Uses the DataFrame index as 
names of parameters (default). Alternatively, a list with parameter
names can be passed as parameter_names. 
"""

class measurement_parameters:
    def __init__(self, parameters, parameter_names=None):
        if parameter_names is None:
            self.parameter_names = parameters.index
        else:
            self.parameter_names = parameter_names
            
        for curr_name, curr_value in zip(
                self.parameter_names, parameters.iloc[:,0]):
            setattr(self, curr_name, curr_value)