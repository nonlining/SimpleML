#-------------------------------------------------------------------------------
# Name:        test Logistic Regression with Neural Network
# Purpose:
#
# Author:      Nonlining
#
# Created:     17/06/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import LogisticRegression as LR
import NeuralNetwork as NN
import pandas as pd
import numpy as np
import string


def main():
    products = pd.read_csv('../Datafiles/amazon_baby.csv')
    #print products.iloc[269]
    #print products['review']
    review_without_punctuation = products['review'].apply(LR.remove_punctuation)

    products['word_count'] = review_without_punctuation.apply(LR.countWords)
    products['clear_words'] = review_without_punctuation
    #print products.iloc[269]['word_count']

    products = products[products['rating'] != 3]
    products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
    np.random.seed(seed=1)

    mask = np.random.rand(len(products)) < 0.8

    train = products[mask]
    test = products[~mask]
    print len(train), len(test)

    reviews = np.array(train['clear_words'])
    labels = np.array(train['sentiment'])

    # using 15000 samples to train
    SentimentWithNN = NN.SentimentNetwork(reviews,labels, learning_rate=0.1)

    SentimentWithNN.train(reviews, labels)

    print "Finish Training"

    # test
    test_reviews = np.array(test['clear_words'])
    test_labels = np.array(test['sentiment'])

    SentimentWithNN.test(test_reviews, test_labels)


if __name__ == '__main__':
    main()
