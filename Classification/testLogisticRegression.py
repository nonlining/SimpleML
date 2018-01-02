#-------------------------------------------------------------------------------
# Name:        test Logistic Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     19/12/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import LogisticRegression as LR
import pandas as pd
import numpy as np

def main():
    np.random.seed(seed=1)
    products = pd.read_csv('../Datafiles/amazon_baby.csv')
    # print 5 items in the list
    print products.head()

if __name__ == '__main__':
    main()
