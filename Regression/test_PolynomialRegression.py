#-------------------------------------------------------------------------------
# Name:        test case for Polynomial Regression
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



def plot(model, features, output):
    import matplotlib.pyplot as pl
    pl.plot(features['power_1'], output, '.', label='output')
    pl.plot(features['power_1'], model.predict(features), '-', label='regression line')
    pl.title('Price vs Squrt ft.')
    pl.ylabel('Price')
    pl.xlabel('Squrt ft.')
    pl.legend(loc='upper left')
    pl.show()




def main():
    # load data from csv files
    train_data = pd.read_csv('../Datafiles/wk3_kc_house_train_data.csv', dtype=dtype_dict)
    test_data = pd.read_csv('../Datafiles/wk3_kc_house_test_data.csv', dtype=dtype_dict)
    validation_data = pd.read_csv('../Datafiles/wk3_kc_house_test_data.csv', dtype=dtype_dict)

    train_data = train_data.sort_values(['sqft_living'])
    test_data = test_data.sort_values(['sqft_living'])
    validation_data = validation_data.sort_values(['sqft_living'])

    poly1_data = PR.polynomial_dataframe(train_data['sqft_living'], 1)
    output1 = train_data['price']
    Model1 = LinearRegression()
    Model1.fit(poly1_data, output1)

    print 'RSS of model 1 ', PR.get_residual_sum_of_squares(Model1, poly1_data, output1)

    plot(Model1, poly1_data, output1)

	# Read more dataset
    houseSet1 = pd.read_csv('../Datafiles/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
    houseSet2 = pd.read_csv('../Datafiles/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
    houseSet3 = pd.read_csv('../Datafiles/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
    houseSet4 = pd.read_csv('../Datafiles/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

    houseSet1 = houseSet1.sort_values(['sqft_living', 'price'])
    houseSet2 = houseSet2.sort_values(['sqft_living', 'price'])
    houseSet3 = houseSet3.sort_values(['sqft_living', 'price'])
    houseSet4 = houseSet4.sort_values(['sqft_living', 'price'])

    polySet1 = PR.polynomial_dataframe(houseSet1['sqft_living'], 15)
    polySet2 = PR.polynomial_dataframe(houseSet2['sqft_living'], 15)
    polySet3 = PR.polynomial_dataframe(houseSet3['sqft_living'], 15)
    polySet4 = PR.polynomial_dataframe(houseSet4['sqft_living'], 15)

    outputSet1 = houseSet1['price']
    outputSet2 = houseSet2['price']
    outputSet3 = houseSet3['price']
    outputSet4 = houseSet4['price']

    modelSet1 = LinearRegression()
    modelSet2 = LinearRegression()
    modelSet3 = LinearRegression()
    modelSet4 = LinearRegression()

    modelSet1.fit(polySet1, outputSet1)
    modelSet2.fit(polySet2, outputSet2)
    modelSet3.fit(polySet3, outputSet3)
    modelSet4.fit(polySet4, outputSet4)

    # Plot order 15 fits
    plot(modelSet1, polySet1, outputSet1)
    plot(modelSet2, polySet2, outputSet2)
    plot(modelSet3, polySet3, outputSet3)
    plot(modelSet4, polySet4, outputSet4)

    # Cross validation to select order
    MAX_POLYNOMIAL_DEGREE = 15
    RSS = np.zeros((MAX_POLYNOMIAL_DEGREE))
    val_output = validation_data['price']
    test_output = test_data['price']




if __name__ == '__main__':
    main()
