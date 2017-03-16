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