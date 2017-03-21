#-------------------------------------------------------------------------------
# Name:        K-nearest Neighbours Regression
# Purpose:
#
# Author:      Nonlining
#
# Created:     16/03/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

def extract(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features
    feature_matrix = np.array(data_frame[features])
    if output == None:
        output_array = []
    else:
        output_array = np.array(data_frame[output])

    return(feature_matrix, output_array)

def normalize_features(matrix):
    norms = np.linalg.norm(matrix, axis = 0)
    features = matrix/norms
    return features, norms

def distance(a, b):
    power2 = (a-b)**2
    dis = np.sqrt(np.sum(power2))
    return dis

def getDistances(all_features, quary_point):
    diff = all_features - quary_point
    distance = np.sqrt(np.sum(diff**2, axis = 1))
    return distance


def FetchKNN(k, features , queries):

    distances = getDistances(features_train, features_query)
    sorted_distances_idx = np.argsort(distances)
    neighbors = sorted_distances_idx[0:k]

    return neighbors