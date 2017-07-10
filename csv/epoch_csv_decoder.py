"""
epoch_csv_decoder.py

This script, when given an epoch .csv files,
imports them into numpy matricies and then,
plots their data, averages their sample, and
plots the average as well
"""
import sys; sys.path.append('..')
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import config


"""
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
"""

samples_per_epoch = config.EPOCH_LENGTH * config.SAMPLING_RATE

def get_sample_data(filename,delim=","):
    """Returns a 2D-numpy array of the sample data from a given file"""
    data = []
    # Read in data from the csv file
    with open("./" + filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                for i in range(len(row)):
                    row[i] = float(row[i])
                data.append(row)

    # Convert the 2D array to a numpy array
    a = np.array(data)

    if len(a) == 0:
        return []

    if len(a) > samples_per_epoch:
        a = a[:samples_per_epoch,:]

    return a

if __name__ == "__main__":

    # List of epoch channel data arrays
    epochs = []

    # Import the data from the csv files provided as arguments
    if len(sys.argv) >= 2:
        for i in range(len(sys.argv)):
            # The first arg is always the script's name
            if i != 0:
                sample_data = get_sample_data(sys.argv[i])
                if len(sample_data) > 1:
                    epochs.append(sample_data)


    # Array to hold the average across epochs
    average = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
    
    for epoch in epochs:
        for i in range(len(epoch)):
            average[i,0:config.NUM_CHANNELS - 1] = average[i,0:config.NUM_CHANNELS - 1] + epoch[i,0:config.NUM_CHANNELS - 1]
  
    if len(epochs) != 0:
        average = average / float(len(epochs))


    # so we are going to have a figure for each channel
    time_between_samples = float(config.EPOCH_LENGTH) / config.SAMPLING_RATE
    for i in range(config.NUM_CHANNELS):
        plt.figure("Channel %d" %(i + 1))
        if config.DISPLAY_AVERAGE:
            plt.plot(np.arange(0.0, float(config.EPOCH_LENGTH), time_between_samples), average[:,i])
            # Plot filtered signal
            t = np.linspace(0, 1.0, 2001)
            x = average[:,i]
            b, a = signal.butter(8, 0.060)
            y = signal.filtfilt(b, a, x)
            plt.plot(np.arange(0.0, float(config.EPOCH_LENGTH), time_between_samples), y)
            
        if config.DISPLAY_TRIALS:
            for epoch in epochs:
            # plot the data for a specific channel for all given epochs
               plt.plot(np.arange(0.0, float(config.EPOCH_LENGTH), time_between_samples)[:len(epoch[:,i])], epoch[:,i])         
    
    
    plt.show()
