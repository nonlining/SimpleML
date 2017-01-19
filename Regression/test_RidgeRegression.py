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
    init_weights = np.array([0.0, 0.0]).reshape((2, 1))
    iterations = 1000
    step = 1e-12
    tolerance = 0

    (train_feature_matrix, train_output) = RR.extract_data_from_features(train_data, features, target)
    (test_feature_matrix, test_output) = RR.extract_data_from_features(test_data, features, target)


if __name__ == '__main__':
    main()
