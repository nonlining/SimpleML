#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Nonlining
#
# Created:     10/01/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import PolynomialRegression as PR
# use sklearn , but I hope can implement this by myself
from sklearn.linear_model import LinearRegression

# Read training and test data
# this map freatures and data types

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




sales = None
train_data = None
test_data = None


# load data from csv files
def loaddata():
    global sales, train_data, test_data
    train_data = pd.read_csv('../Datafiles/kc_house_train_data.csv')
    test_data = pd.read_csv('../Datafiles/kc_house_test_data.csv')


def main():
    pass

if __name__ == '__main__':
    main()
