# -*- coding: utf-8 -*-
""" xdf2mne.py
    ==========

Overview:
            
    loads a .xdf file recorded using LSL, and writes out a .fif file readable
    by the Python MEG & EEG analysis package (MNE) that contains the same
    information.
    
Example:
    
    $ python xdf2mne.py
    
.. _Google Python Style Guide:
http://google.github.io/styleguide/pyguide.html
        
    Copyright 2017 by
    Brent Lance <brent.j.lance.civ@mail.mil>
    and the United States Army Research Laboratory
    All rights reserved.    
"""

import os.path as path
import numpy as np
import mne
from eeg_event_codes import *


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from mne.viz import tight_layout

import pickle
import config
import argparse

parser = argparse.ArgumentParser(description='Builds a classifier from raw EEG data using the MNE library')
parser.add_argument('-p','--plot',
                        dest="plot_epochs",
                        action='store_true',
                        default=False,
                        help='Displays a plot of the avarage target and non-target epoch data as well as their diffrence')
args = parser.parse_args()

#import data
data_path = 'C:\Users\shijb\Desktop\P300Project\csv'
data_file = 'PRD.csv'

EEG_data = np.genfromtxt(data_path + '\\' + data_file, delimiter=",")
EEG_labels = ['time_stamps', 'event', 'Cz', 'Pz', 'Oz', 'O1', 'O2']
EEG_types = ['misc', 'stim', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

sample_rate = 250
info = mne.create_info(EEG_labels, sample_rate, ch_types=EEG_types)

raw_data = mne.io.RawArray(np.transpose(EEG_data), info)

filtered_data = raw_data.filter(0.1, h_freq=None, l_trans_bandwidth = 0.05)
del raw_data
filtered_data = filtered_data.filter( l_freq=None, h_freq=40)




events = mne.find_events(filtered_data, stim_channel='event', shortest_event=1, consecutive=True)
filtered_data.drop_channels(['event'])

event_id = {
    #'non-target/row0':0,
    'nontarget/row1':1,
    'nontarget/row2':2,
    'nontarget/row3':3,
    'nontarget/row4':4,
    'nontarget/row5':5,
    'nontarget/col0':8,
    'nontarget/col1':9,
    'nontarget/col2':10,
    'nontarget/col3':11,
    'nontarget/col4':12,
    'nontarget/col5':13,
    #'target/row0':16,
    'target/row1':17,
    'target/row2':18,
    'target/row3':19,
    'target/row4':20,
    'target/row5':21,
    'target/col0':24,
    'target/col1':25,
    'target/col2':26,
    'target/col3':27,
    #'target/col4':28,
    'target/col5':29,
    'none':32
    }



epochs = mne.Epochs(filtered_data, events, event_id=event_id, add_eeg_ref=False, tmin = 0, tmax=1, decim=2)
epochs.load_data()

X = []
y = []

# Creating example cases for training/classification
X_target = epochs['target'].pick_channels(['Cz']).get_data()
X_nontarget = epochs['nontarget'].pick_channels(['Cz']).get_data()

print np.shape(X_target)
print np.shape(X_nontarget)

for ep in X_target:
    ep =  ep[:,0:-1]
    X.append(np.ravel(ep))
    y.append(1)

for ep in X_nontarget:
    ep =  ep[:,0:-1]
    X.append(np.ravel(ep))
    y.append(0)

print np.shape(X)
print np.shape(y)

X = np.matrix(X)
y = np.array(y)

clf1 = svm.SVC(kernel='poly', probability=True)
clf2 = svm.SVC(kernel='poly', degree=2, probability=True)
clf3 = svm.SVC(probability=True)
eclf1 = VotingClassifier(estimators=[('deg3', clf1), ('quad', clf2), ('rbf', clf3)], voting='soft')
#eclf1 = #eclf1.fit(np.matrix(X),np.array(y))


classifier = clf3

scores = cross_val_score(classifier, X, y, cv=2)
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
classifier.fit(X,y)

outfile = open("mne_classifier.pkl", 'wb+')
pickle.dump(classifier, outfile)
outfile.close()



if args.plot_epochs:
    picks = mne.pick_types(filtered_data.info, eeg=True, stim=False, eog=False)
    epochs_xd = mne.Epochs(filtered_data, events, event_id=event_id, add_eeg_ref=False, tmin = 0, tmax=1, picks=picks, baseline=None)
    evoked = epochs.average()
    picks = mne.pick_channels(evoked.ch_names, ['O1'])
    evoked_target = epochs['target'].average()
    evoked_nontarget = epochs['nontarget'].average()
    difference = mne.combine_evoked([evoked_target, evoked_nontarget], [-1, 1])
    mne.viz.plot_compare_evokeds({'target':evoked_target, 'nontarget':evoked_nontarget, 'difference':difference}, picks=picks)
