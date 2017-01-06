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
import LinearRegression as SR

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

    Model1_features = ['sqft_living']
    Model1_output = ['price']
    (feature_matrix1, output_vector) = SR.get_data(train_data, Model1_features, Model1_output)
    (feature_matrix1_t, output_vector_t) = SR.get_data(test_data, Model1_features, Model1_output)

    step_size1 = 7.0e-12
    tolerance1 = 2.5e7
    init_weights1 = np.array([-47000.0, 1.0]).reshape((2, 1))

    Model1_weights = SR.regression_gradient_descent(feature_matrix1,
                                                     output_vector,
                                                     init_weights1,
                                                     step_size1,
                                                     tolerance1)


    test1_predictions = SR.predict(feature_matrix1_t, Model1_weights)
    print test1_predictions[0]
    RSS1 = SR.get_residual_sum_of_squares(feature_matrix1_t, Model1_weights, output_vector_t)
    print RSS1



if __name__ == '__main__':
    main()