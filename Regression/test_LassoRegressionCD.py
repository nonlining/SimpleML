#-------------------------------------------------------------------------------
# Name:        test Lasso Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     09/02/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import LassoRegression as LR
from math import sqrt
import math

def main():

    all_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                    'floors', 'waterfront', 'view', 'condition', 'grade',
                    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
                  'sqft_living15':float, 'grade':int, 'yr_renovated':int,
                  'price':float, 'bedrooms':float, 'zipcode':str,
                  'long':float, 'sqft_lot15':float, 'sqft_living':float,
                  'floors':float, 'condition':int, 'lat':float, 'date':str,
                  'sqft_basement':int, 'yr_built':int, 'id':str,
                  'sqft_lot':int, 'view':int}

    sales = pd.read_csv('../Datafiles/kc_house_data.csv', dtype=dtype_dict)
    train_data = pd.read_csv('../Datafiles/kc_house_train_data.csv', dtype=dtype_dict)
    test_data = pd.read_csv('../Datafiles/kc_house_test_data.csv', dtype=dtype_dict)

    simple_features = ['sqft_living', 'bedrooms']
    my_output = 'price'


    (simple_feature_matrix, output) = LR.extract(sales, simple_features, my_output)
    print output
    simple_feature_matrix, norms = LR.normalize_features(simple_feature_matrix)
    print simple_feature_matrix[:, 1]


    weights = np.array([1., 4., 1.])
    R1 = LR.getRo(simple_feature_matrix, output, weights, 1)
    R2 = LR.getRo(simple_feature_matrix, output, weights, 2)
    print "R1 is :",R1
    print "R2 is :",R2
    print LR.lasso_coordinate_descent_step(1, np.array([[3./sqrt(13),1./sqrt(10)],[2./sqrt(13),3./sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1)

    initial_weights = np.zeros(3)
    l1_penalty = 1e7
    tolerance = 1.0
    (simple_feature_matrix, output) = LR.extract(sales, simple_features, my_output)
    (normalized_simple_feature_matrix, simple_norms) = LR.normalize_features(simple_feature_matrix)
    weights = LR.lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)

    pred = LR.predict(normalized_simple_feature_matrix, weights)
    residuals = pred - output
    rss_normalized_data = (residuals * residuals).sum()
    print rss_normalized_data

    print weights



    (feature_matrix, output) = LR.extract(train_data, all_features, my_output)

    (normalized_feature_matrix, norms) = LR.normalize_features(feature_matrix)

    (all_feature_matrix, train_output) = LR.extract(train_data, all_features, my_output)
    (normalized_all_feature_matrix, train_norms) = LR.normalize_features(all_feature_matrix)
    initial_weights = np.zeros(len(all_features) + 1)


    l1_penalty=1e4
    tolerance=5e5
    weights1e4 = LR.lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, train_output, initial_weights, 1e4, tolerance)
    weights1e4_normalized = weights1e4 / train_norms
    print weights1e4_normalized



if __name__ == '__main__':
    main()