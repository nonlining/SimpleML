#-------------------------------------------------------------------------------
# Name:        polynomial regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     09/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd

def polynomial_dataframe(feature, degree):

    poly = pd.DataFrame()

    poly['power_1'] = feature

    if degree > 1:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            poly[name] = feature.apply(lambda x: x**power)

    return poly