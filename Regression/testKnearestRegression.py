#-------------------------------------------------------------------------------
# Name:        test Knearest Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     23/03/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import KnearestRegression as KNN

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

feature_list = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']

def main():
    train = pd.read_csv('../Datafiles/kc_house_data_small_train.csv', dtype=dtype_dict)
    valid = pd.read_csv('../Datafiles/kc_house_data_validation.csv', dtype=dtype_dict)
    test = pd.read_csv('../Datafiles/kc_house_data_small_test.csv', dtype=dtype_dict)


    features_train, output_train = KNN.extract(train, feature_list, 'price')
    features_test, output_test = KNN.extract(test, feature_list, 'price')
    features_valid, output_valid = KNN.extract(valid, feature_list, 'price')


    features_train, norms = KNN.normalize_features(features_train) # normalize training set features (columns)
    features_test = features_test / norms # normalize test set by training set norms
    features_valid = features_valid / norms # normalize validation set by training set norms
    print features_test[0]
    print features_train[9]

    print KNN.distance(features_test[0], features_train[9])

    min_dist = float('inf')
    min_index = 0

    for i, j in enumerate(features_train[0:10]):
        dist = KNN.distance(features_test[0], j)
        print i, dist
        if dist < min_dist:
            min_dist = dist
            min_index = i

    print min_index, min_dist









if __name__ == '__main__':
    main()
