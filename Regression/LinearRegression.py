#-------------------------------------------------------------------------------
# Name:        Linear Regression 
# Purpose:
#
# Author:      Nonlining
#
# Created:     05/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from math import log


def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    pred_y = model.predict(data)
    # Then compute the residuals/errors
    Residual = outcome - pred_y

    # print Residual
    # Then square and add them up
    RSS = (Residual*Residual).sum()
    return(RSS)



def Linear_Regression(train_data, target, features):
    pass

