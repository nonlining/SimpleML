#-------------------------------------------------------------------------------
# Name:        test Ridge Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     19/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import RidgeRegression as RR

import matplotlib.pyplot as plt

# Read training and test data
dtype_dict = {'bathrooms':float, 'waterfront':int,
              'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float,
              'floors':str, 'condition':int,
              'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int,
              'id':str, 'sqft_lot':int, 'view':int}


def main():
    train_data = pd.read_csv('../Datafiles/kc_house_train_data.csv', dtype=dtype_dict)
    test_data = pd.read_csv('../Datafiles/kc_house_test_data.csv', dtype=dtype_dict)


    features = ['sqft_living']
    target = ['price']

    initial_weights = np.array([0.0, 0.0]).reshape((2, 1))
    iterations = 1000
    stepSize = 1e-12
    tolerance = 0
    train_feature, train_output = RR.extract_data_from_features(train_data, features, target)
    test_feature ,  test_output = RR.extract_data_from_features(test_data, features, target)
    l2_penalty = 0

    simple_model = RR.Ridge_Regression_gradient_descent(train_feature,
                                                         train_output,
                                                      initial_weights,
                                                   step_size=stepSize,
                                                l2_penalty=l2_penalty,
                                                  tolerance=tolerance,
                                             max_iterations=iterations)
    print simple_model

    l2_penalty = 1e11
    simple_model_l2 = RR.Ridge_Regression_gradient_descent(train_feature,
                                                            train_output,
                                                         initial_weights,
                                                      step_size=stepSize,
                                                   l2_penalty=l2_penalty,
                                                     tolerance=tolerance,
                                                max_iterations=iterations)
    print simple_model_l2


    plt.title('L2 penalty comparison')
    plt.ylabel('Price')
    plt.xlabel('Sq.ft.')
    plt.plot(train_feature[:,1], train_output, 'k.', label='training data')
    plt.plot(train_feature[:,1], RR.predict(train_feature, simple_model), 'b-', label='L2=0')
    plt.plot(train_feature[:,1], RR.predict(train_feature, simple_model_l2), 'r-', label='L2=1e11')
    plt.legend(loc='upper left')

    plt.show()

    simple_model_test_RSS = RR.get_residual_sum_of_squares(test_feature,
                                                        initial_weights,
                                                            test_output)

    simple_model_test_RSS_noL2 = RR.get_residual_sum_of_squares(test_feature,
                                                         simple_model,
                                                         test_output)

    simple_model_test_RSS_L2 = RR.get_residual_sum_of_squares(test_feature,
                                                         simple_model_l2,
                                                         test_output)

    # Complex model - two features
    features = ['sqft_living', 'sqft_living15']
    target = ['price']
    initial_weights = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
    iterations = 1000
    step = 1e-12
    tolerance = 0

    train_feature, train_output = RR.extract_data_from_features(train_data, features, target)
    test_feature, test_output = RR.extract_data_from_features(test_data, features, target)

    l2_penalty = 0
    complex_model = RR.Ridge_Regression_gradient_descent(train_feature,
                        train_output,
                        initial_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

    l2_penalty = 1e11
    complex_model_l2 = RR.Ridge_Regression_gradient_descent(train_feature,
                        train_output,
                        initial_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

    complex_model_test_RSS = RR.get_residual_sum_of_squares(test_feature,
                                                         initial_weights,
                                                             test_output)
    print complex_model_test_RSS

    complex_model_test_RSS_noL2 = RR.get_residual_sum_of_squares(test_feature,
                                                                complex_model,
                                                                  test_output)
    print complex_model_test_RSS_noL2

    complex_model_test_RSS_L2 = RR.get_residual_sum_of_squares(test_feature,
                                                           complex_model_l2,
                                                                test_output)
    print complex_model_test_RSS_L2



if __name__ == '__main__':
    main()
