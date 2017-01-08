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

	# Model 1 features
    Model1_features = ['sqft_living']
    Model1_output = ['price']

    feature_matrix1, output_vector1 = SR.extract_data_from_features(train_data,
                                                                   Model1_features,
                                                                   Model1_output)


    feature_matrix1_targets, output_vector1_targets = SR.extract_data_from_features(
	                                                                   test_data,
                                                                       Model1_features,
                                                                       Model1_output)

    step_size1 = 7.0e-12
    tolerance1 = 2.5e7
    # you get set your init weights for this question, but it will take some time to train
    init_weights1 = np.array([-47000.0, 1.0]).reshape((2, 1))

    # Model 1 training
    Model1_weights = SR.regression_gradient_descent(feature_matrix1,
                                                    output_vector1,
                                                    init_weights1,
                                                    step_size1,
                                                    tolerance1)

    test1_predictions = SR.predict(feature_matrix1_targets, Model1_weights)
    print "The first house prediction price of test data", test1_predictions[0]
    # Model 1 RSS
    RSS1 = SR.get_residual_sum_of_squares(feature_matrix1_targets, Model1_weights, output_vector1_targets)
    print "RSS of model 1 ",RSS1



	# Model 2 features

    Model2_features = ['sqft_living', 'sqft_living15']
    Model2_output = ['price']
	# extract matrix from training data correspond to features and output
    feature_matrix2, output_vector2 = SR.extract_data_from_features(train_data,
                                                                   Model2_features,
                                                                   Model2_output)

    feature_matrix2_targets, output_vector2_targets = SR.extract_data_from_features(
	                                                                   test_data,
                                                                       Model2_features,
                                                                       Model2_output)

    # Set parameters
    step_size2 = 4.0e-12
    tolerance2 = 1.0e9
    init_weights2 = np.array([-100000.0, 1.0, 1.0]).reshape((3, 1))

    # Model 2 training
    Model2_weights = SR.regression_gradient_descent(feature_matrix2,
                                                    output_vector2,
                                                    init_weights2,
                                                    step_size2,
                                                    tolerance2)


    test2_predictions = SR.predict(feature_matrix2_targets, Model2_weights)
    print "The first house prediction price of test data", test2_predictions[0]
    RSS2 = SR.get_residual_sum_of_squares(feature_matrix2_targets, Model2_weights, output_vector2_targets)
    print "RSS of model 2 ",RSS2


if __name__ == '__main__':
    main()