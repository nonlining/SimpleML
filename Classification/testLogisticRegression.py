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
import LogisticRegression as LR
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import string

def main():
    products = pd.read_csv('../Datafiles/amazon_baby.csv')
    print products.iloc[269]
    print products['review']
    review_without_punctuation = products['review'].apply(LR.remove_punctuation)
    products['word_count'] = review_without_punctuation.apply(LR.countWords)
    products['clear_words'] = review_without_punctuation
    print products.iloc[269]['word_count']



if __name__ == '__main__':
    main()
