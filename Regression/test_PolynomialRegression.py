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

train_data = None
test_data = None
validation_data = None


def plot(model, features, output):
    import matplotlib.pyplot as pl
    pl.plot(features['power_1'], output, '.', label='output')
    pl.plot(features['power_1'], model.predict(features), '-', label='regression line')
    pl.title('Price vs Sq.ft.')
    pl.ylabel('Price')
    pl.xlabel('Sq.ft.')
    pl.legend(loc='upper left')
    pl.show()

# load data from csv files
def loaddata():
    global train_data, test_data, validation_data
    train_data = pd.read_csv('../Datafiles/wk3_kc_house_train_data.csv', dtype=dtype_dict)
    test_data = pd.read_csv('../Datafiles/wk3_kc_house_test_data.csv', dtype=dtype_dict)
    validation_data = pd.read_csv('../Datafiles/wk3_kc_house_test_data.csv', dtype=dtype_dict)


def main():
    loaddata()
    global train_data, test_data, validation_data

    train_data = train_data.sort(['sqft_living'])
    test_data = test_data.sort(['sqft_living'])
    validation_data = validation_data.sort(['sqft_living'])

    poly1_data = PR.polynomial_dataframe(train_data['sqft_living'], 1)\

    output = train_data['price']


    model1 = LinearRegression()
    model1.fit(poly1_data, output)

    plot(model1, poly1_data, output)


if __name__ == '__main__':
    main()
