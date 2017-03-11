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

def extract(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features
    feature_matrix = np.array(data_frame[features])
    if output == None:
        output_array = []
    else:
        output_array = np.array(data_frame[output])

    return(feature_matrix, output_array)



def normalize_features(matrix):
    norms = np.linalg.norm(matrix, axis = 0)
    features = matrix/norms
    return features, norms


def predict(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return(predictions)


def RSS(feature_matrix, weights, output):

    predictions = predict(feature_matrix, weights)
    residual = np.sum((predictions - output) ** 2)
    return(residual)

def getRo(feature_matrix, output, weights, i):

    prediction = predict(feature_matrix, weights)
    feature_i = feature_matrix[:, i]
    ro_i = (feature_i * (output - prediction + weights[i] * feature_i)).sum()
    return ro_i


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    rho = getRo(feature_matrix, output, weights, i)

    if i==0:
        new_weight = rho
    elif rho < (-l1_penalty/2.0):
        new_weight = rho + (l1_penalty/2.0)
    elif rho > (l1_penalty/2.0):
        new_weight = rho - (l1_penalty/2.0)
    else:
        new_weight = 0.0
    return new_weight


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):

    D = feature_matrix.shape[1]
    weights = initial_weights[:]
    change = np.zeros(initial_weights.shape)
    max_change = tolerance

    while max_change >= tolerance:
        for idx in range(D):
            new_weight = lasso_coordinate_descent_step(idx, feature_matrix,
                                                       output, weights,
                                                       l1_penalty)
            change[idx] = np.abs(new_weight - weights[idx])
            weights[idx] = new_weight
        max_change = max(change)

    return weights