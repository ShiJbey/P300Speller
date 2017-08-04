"""
Uses P300 Speller epoch data in .csv files to train
a classifier for use when preocessing live EEG data
"""
import csv
import sys
import os
import math
import pickle
import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import config
sys.path.append('..')

def down_sample_data(data, current_sample_rate, target_sample_rate=128):
    """
    Down samples a given data sample by extracting a subset of the data
    Params:
        data - the data that will be down sampled
        target_sample_rate - the rate, in Hz, that this data should be down sampled to
        current_sample_rate - the rate, in Hz, that this data was sampled at
    Returns:
        resampled_data - numpy array of the data had it been sampled at the target rate
    """
    if target_sample_rate > current_sample_rate:
        raise ArithmeticError("Cannot down sample to a higher sample rate")
    
    if len(data) == 0:
        return data
    
    samples_to_skip = int(math.ceil(float(current_sample_rate) / target_sample_rate))
    
    sample_index = 0

    resampled_data = np.zeros((0,np.shape(data)[1]))
    
    while sample_index < len(data):
        #print "Adding sample num %d" % sample_index
        resampled_data = np.vstack((resampled_data, data[sample_index,:]))
        sample_index += samples_to_skip

    return resampled_data


def create_example_cases(average_data, p300_index):
    """
    Given a list of averaged epoch data matrices and index indicating the
    row/col of the expected P300, outputs a set of example cases and classifications
    """
    X = []
    y = []

    for index in range(len(average_data)):
        data = np.ravel(average_data[index])
        X = np.vstack((X, data))
        if index == p300_index:
            y = np.append(y, 1)
        else:
            y = np.append(y, 0)
    
    return X, y


def get_example_case(filepath):
    """Returns a 2D-numpy array of the sample data from a given file"""
    data = np.zeros((0, len(config.EEG_CHANNELS)))
    classification = -1

    # Read in data from the csv file
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        classification = reader.next()[0]
        for row in reader:
            for i in range(len(row)):
                row[i] = float(row[i])
                
                row = np.array(row)
            data = np.vstack((data, row))
    
    data = np.ravel(data)

    return data, classification



if __name__ == '__main__':
    # Use the directory passed as a param to open all the files epoch files
    print sys.argv
    directory = sys.argv[1]

    # Holds all of the training examples
    X = np.zeros((0, num_samples_possible * len(config.EEG_CHANNELS)))
    # Holds all of the example classifications
    y = np.array([])

    

    data_filenames = os.listdir(directory)
    for fname in data_filenames:
        if fname.contains("RawData"):
            data_filenames.remove(fname)

    # Generate example set from all of the files in the directory
    for name in sorted(data_filenames):
        example_data, example_class = get_example_case(directory + name)
        if example_class != -1:
            X = np.vstack((X, example_data))
            y = np.append(y, example_class)
    

    # Feature scale X
    X = preprocessing.scale(X)

    print "Training classifiers."
    clf1 = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    clf2 = svm.SVC(kernel='poly', degree=2, class_weight='balanced', probability=True)
    clf3 = svm.SVC(class_weight='balanced', probability=True)
    eclf1 = VotingClassifier(estimators=[('lin', clf1), ('quad', clf2), ('rbf', clf3)], voting='soft')
    eclf1 = eclf1.fit(X,y)
    print "Done."

    classifier = eclf1

    scores = cross_val_score(classifier, X, y, cv=10)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


    outfile = open(config.CLASSIFIER_FILENAME, 'wb')
    pickle.dump(classifier, outfile)
    outfile.close()
    
    print "Exported classifier to: %s" % config.CLASSIFIER_FILENAME
    