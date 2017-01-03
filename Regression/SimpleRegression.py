#-------------------------------------------------------------------------------
# Name:        Simple Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     02/01/2017
# Copyright:   (c) Nonlining 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    sum_of_input = input_feature.sum()
    sum_of_output = output.sum()


    # compute the product of the output and the input_feature and its sum
    sum_product_of_in_out = (input_feature*output).sum()

    n = len(input_feature)


    # compute the squared value of the input_feature and its sum
    sum_squared_input = (input_feature * input_feature).sum()

    # use the formula for the slope
    numerator = sum_product_of_in_out - (sum_of_input*sum_of_output)/float(n)
    denominator = sum_squared_input - (sum_of_input*sum_of_input)/float(n)

    slope =  numerator/denominator

    # use the formula for the intercept

    intercept = float(sum_of_output)/n - slope*float(sum_of_input)/n

    return(intercept, slope)


def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept)/float(slope)

    return estimated_feature


def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:

    predicted_values = intercept + slope*input_feature

    return predicted_values

def get_residual_sum_of_squares(input_feature, output, intercept,slope):

    gap = (output - (intercept + slope*input_feature))
    RSS = (gap*gap).sum()
    return(RSS)








