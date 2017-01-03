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
import SimpleRegression as SR

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

    intercept, slope = SR.simple_linear_regression(feature_sqft_living, outputs)




if __name__ == '__main__':
    main()