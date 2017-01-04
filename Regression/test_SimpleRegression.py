#-------------------------------------------------------------------------------
# Name:        test Simple Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     03/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import SimpleRegression as SR

sales = None
train_data = None
test_data = None


# load data from csv files
def loaddata():
    global sales, train_data, test_data
    train_data = pd.read_csv('../Datafiles/kc_house_train_data.csv')
    test_data = pd.read_csv('../Datafiles/kc_house_test_data.csv')


def main():
    loaddata()
    feature_sqft_living = np.array(train_data['sqft_living'])
    feature_bedrooms = np.array(train_data['bedrooms'])
    outputs = np.array(train_data['price'])
    x = [1,2,3,4,5]
    y = [2,1,4,3,5]

    intercept, slope = SR.simple_linear_regression(np.array(x), np.array(y))
    print intercept, slope
    print SR.get_regression_predictions(1, slope = slope, intercept = intercept)


    intercept, slope = SR.simple_linear_regression(feature_sqft_living, outputs)
    # predict 2650 sqft
    pred_2650sqft =  SR.get_regression_predictions(2650, intercept, slope)
    print pred_2650sqft
    # RSS of train data
    RSS_train_data = SR.get_residual_sum_of_squares(feature_sqft_living, outputs, intercept, slope)
    print RSS_train_data
    # hosue price is $800000 and its squre feet
    sqrtft_800000 = SR.inverse_regression_predictions(800000, slope = slope , intercept = intercept)
    print sqrtft_800000





if __name__ == '__main__':
    main()