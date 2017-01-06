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


def get_residual_sum_of_squares(features_matrix, weights, outputs):
    pred_y = predict(features_matrix, weights)
    Residual = outputs - pred_y
    RSS = (Residual*Residual).sum()
    return(RSS)

def get_data(data_frame, features, output):
    data_frame['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = np.array(data_frame[features])

    if output != None:
        output_array = np.array(data_frame[output])
    else:
        output_array = []

    return(features_matrix, output_array)

def predict(feature_matrix, weights):
    return np.dot(feature_matrix, weights)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)

    while not converged:
        predictions = predict(feature_matrix, weights)
        errors =  output - predictions
        partial =  -2*np.dot(feature_matrix.T, errors)
        weights = weights - step_size*partial
        gradient_magnitude = np.sqrt(np.sum(partial*partial))
        if gradient_magnitude < tolerance:
            converged = True

    return(weights)

