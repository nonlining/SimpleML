#-------------------------------------------------------------------------------
# Name:        Ridge Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     17/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np


def predict(features, weights):
    return np.dot(features ,weights)


def Ridge_Regression_gradient_descent(features, output, initial_weights, step_size, l2_penalty, tolerance, max_iterations=100):
    converged = False
    weights = np.array(initial_weights)
    weights_scaler = np.ones(len(weights))
    weights_scaler[1:] = 1.0 - 2.0 * step_size * l2_penalty
    weights_scaler = weights_scaler.reshape((len(weights),1))
    iteration = 0
    print 'Starting Gradient descent'
    while not converged:
        prediction = predict(features, weights)
        errors = output - prediction
        product = (features.T).dot(errors)
        gradient = -2.0 * product + 2.0 * l2_penalty * weights
        gradient_magnitude = np.sqrt(np.sum(gradient * gradient))
        weights = weights_scaler * weights + 2.0 * step_size * product
        iteration += 1
        if (iteration > max_iterations) or (gradient_magnitude < tolerance):
            converged = True
            print 'Stopping at iteration: ' + str(iteration - 1)
    return(weights)
