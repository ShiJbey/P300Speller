"""
p300_classifier.py

Uses EEG data and scikit-learn to produce
a support vector classifier for the p300 speller
"""
import pickle
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    