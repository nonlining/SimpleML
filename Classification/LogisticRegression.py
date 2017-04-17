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