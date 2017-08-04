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
import eeg_event_codes


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

parser.add_argument('classifier_path',
                    type=str,
                    nargs=1,
                    help='Specifies a path for the raw data for making classifier')

parser.add_argument('-t','--train',
                    dest="train_classifier",
                    action='store_true',
                    default=False,
                    help='Option to train and export classifier')    

parser.add_argument('-p','--plot',
                    dest="plot_epochs",
                    action='store_true',
                    default=False,
                    help='Displays a plot of the avarage target and non-target epoch data as well as their diffrence')

args = parser.parse_args()


data_path = args.classifier_path[0]
EEG_data = np.genfromtxt(data_path, delimiter=",")
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





epochs = mne.Epochs(filtered_data, events, event_id=eeg_event_codes.EVENT_ID, add_eeg_ref=False, tmin = 0, tmax=1, decim=config.DOWN_SAMPLE_FACTOR)
epochs.load_data()


if args.train_classifier:

    X = []
    y = []

    # Creating example cases for training/classification
    X_target = epochs['target'].pick_channels(config.CHANNELS_FOR_SVC).get_data()
    X_nontarget = epochs['nontarget'].pick_channels(config.CHANNELS_FOR_SVC).get_data()

    for ep in X_target:
        ep =  ep[:,0:]
        X.append(np.ravel(ep))
        y.append(1)

    for ep in X_nontarget:
        ep =  ep[:,0:]
        X.append(np.ravel(ep))
        y.append(0)

    X = np.matrix(X)
    y = np.array(y)

    clf1 = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    clf2 = svm.SVC(kernel='poly', degree=2, class_weight='balanced', probability=True)
    clf3 = svm.SVC(class_weight='balanced', probability=True)
    eclf1 = VotingClassifier(estimators=[('deg3', clf1), ('quad', clf2), ('rbf', clf3)], voting='soft')#eclf1 = #eclf1.fit(np.matrix(X),np.array(y))


    classifier = eclf1

    print "Cross Validating Ensemble"
    scores = cross_val_score(classifier, X, y, cv=2)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    print "Training classifier"
    classifier.fit(X,y)
    print "Done"

    print "Exporting classifiers"
    outfile = open("mne_classifier.pkl", 'wb+')
    pickle.dump(classifier, outfile)
    outfile.close()
    print "Done"

if args.plot_epochs:
    picks = mne.pick_types(filtered_data.info, eeg=True, stim=False, eog=False)
    evoked = epochs.average()
    picks = mne.pick_channels(evoked.ch_names, ['O1'])
    evoked_target = epochs['target'].average()
    evoked_nontarget = epochs['nontarget'].average()
    difference = mne.combine_evoked([evoked_target, evoked_nontarget], [-1, 1])
    mne.viz.plot_compare_evokeds({'target':evoked_target, 'nontarget':evoked_nontarget, 'difference':difference}, picks=picks)
