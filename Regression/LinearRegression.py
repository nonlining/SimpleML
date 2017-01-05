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

def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2*np.dot(errors, feature)
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array

    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            partial =  feature_derivative(errors,feature_matrix[:,i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += partial*partial
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size*partial
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return(weights)
	

def Linear_Regression(train_data, target, features):
    pass

