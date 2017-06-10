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


def extract(data_frame, features, label):

    data_frame['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = data_frame[features].to_numpy()
    if label != None:
        label_array = data_frame[label].to_numpy()
    else:
        label_array = []
		
    return(features_matrix, label_array)

def remove_punctuation(text):
    if type(text) is str:
        return text.translate(None, string.punctuation)
    else:
        return str(text).translate(None, string.punctuation)


def countWords(string):
    wordCount = {}
    for i in string.split(' '):
        if i in wordCount:
            wordCount[i] += 1
        else:
            wordCount[i] = 1

    return wordCount

def get_classification_accuracy(model, data, true_labels):
    pred_result = model.predict(data)
    correct = sum(pred_result == true_labels)

    accuracy = correct/float(len(true_labels))
    return accuracy

def compute_probability(score):

    return 1.0/(1 + np.exp(-score))

