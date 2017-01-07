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

def extract_data_from_features(data, features, output):
    data['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = np.array(data[features])

    if output != None:
        output_array = np.array(data[output])
    else:
        output_array = []

    return(features_matrix, output_array)

def predict(feature_matrix, weights):
    return np.dot(feature_matrix, weights)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    #count = 0

    while not converged:
        predictions = predict(feature_matrix, weights)
        errors =  output - predictions
        partial =  -2*np.dot(feature_matrix.T, errors)
        weights = weights - step_size*partial
        gradient_magnitude = np.sqrt(np.sum(partial*partial))
        #count = count + 1
        #if (count%10000 == 0):
        #    print gradient_magnitude

        if gradient_magnitude < tolerance:
            converged = True

    return(weights)

