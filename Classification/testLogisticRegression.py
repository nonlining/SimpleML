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

    review_without_punctuation = products['review'].apply(LR.remove_punctuation)
    products['word_count'] = review_without_punctuation.apply(LR.countWords)
    products['clear_words'] = review_without_punctuation

    # print 5 items in the list
    print review_without_punctuation.head()

    # remove all rating 3 , they are considered as neutral
    products = products[products['rating'] != 3]
    products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

    mask = np.random.rand(len(products)) < 0.8



if __name__ == '__main__':
    main()
