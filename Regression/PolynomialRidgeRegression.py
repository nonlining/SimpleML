#-------------------------------------------------------------------------------
# Name:        Polynomial Ridge Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     01/02/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

def polynomial_dataframe(feature, degree):

    poly = pd.DataFrame()

    poly['power_1'] = feature

    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly[name] = feature.apply(lambda x: x**power)

    return poly