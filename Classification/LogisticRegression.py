#-------------------------------------------------------------------------------
# Name:        Logistic Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     17/04/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import string

def remove_punctuation(text):
    return text.translate(None, string.punctuation)



def get_classification_accuracy(model, data, true_labels):

    pred_result = model.predict(data)
    correctPredictions = sum(pred_result == true_labels)

    accuracy = correctPredictions/float(len(true_labels))

    return accuracy

