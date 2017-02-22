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

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

all_features = ['bedrooms', 'bedrooms_square', 'bathrooms',
                'sqft_living', 'sqft_living_sqrt', 'sqft_lot',
                'sqft_lot_sqrt', 'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']


def main():
    sales = pd.read_csv('../Datafiles/kc_house_data.csv', dtype=dtype_dict)
    # only 2 features case
    simple_features = ['sqft_living', 'bedrooms']
    my_output = 'price'
    (simple_feature_matrix, output) = LR.extract(sales, simple_features, my_output)
    print simple_feature_matrix
    print output
    simple_feature_matrix, norms = LR.normalize_features(simple_feature_matrix)
    print simple_feature_matrix[:, 1]


    weights = np.array([1., 4., 1.])
    R1 = LR.getRo(simple_feature_matrix, output, weights, 1)
    R2 = LR.getRo(simple_feature_matrix, output, weights, 2)
    print "R1 is :",R1
    print "R2 is :",R2





if __name__ == '__main__':
    main()