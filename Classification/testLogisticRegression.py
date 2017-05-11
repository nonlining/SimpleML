#-------------------------------------------------------------------------------
# Name:        test Logistic Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     17/04/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import sframe as sf
import LogisticRegression as LR


def main():
    products = sf.SFrame('../Datafiles/amazon_baby.gl/')
    print products
    print products[269]
    review_without_punctuation = products['review'].apply(LR.remove_punctuation)
    products = products[products['rating'] != 3]
    print len(products)

if __name__ == '__main__':
    main()
