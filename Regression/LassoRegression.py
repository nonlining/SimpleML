#-------------------------------------------------------------------------------
# Name:        Lasso Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     26/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    predictions = feature_matrix.dot(weights)
    rho = (feature_matrix[:, i].T).dot(output - predictions + (weights[i] * feature_matrix[:, i]))
    if i==0:
        new_weight = rho
    elif rho < (-l1_penalty/2.0):
        new_weight = rho + (l1_penalty/2.0)
    elif rho > (l1_penalty/2.0):
        new_weight = rho - (l1_penalty/2.0)
    else:
        new_weight = 0.0
    return new_weight