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


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):

    D = feature_matrix.shape[1]
    weights = copy.copy(initial_weights)
    change = np.zeros(initial_weights.shape)
    converged = False

    while not converged:
        # Evaluate over all features
        for idx in range(D):
            # New weight for feature
            new_weight = lasso_coordinate_descent_step(idx, feature_matrix,
                                                       output, weights,
                                                       l1_penalty)
            # Compute change in weight for feature
            change[idx] = np.abs(new_weight - weights[idx])
            # assign new weight
            weights[idx] = new_weight
        # Maximum change in weight, after all changes have been computed
        max_change = max(change)
        if max_change < tolerance:
            converged = True
    return weights