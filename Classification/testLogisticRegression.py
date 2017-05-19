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
    products['word_count'] = review_without_punctuation.apply(LR.countWords)

    print products[269]['word_count']

    products = products[products['rating'] != 3]
    print len(products)

    products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

    print products

    train_data, test_data = products.random_split(.8, seed=1)

    print len(train_data)
    print len(test_data)




if __name__ == '__main__':
    main()
