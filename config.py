"""
Constants used in running the p300_speller module
or any of its associated modules
"""
import math

#======================================================#
#               DATA/TRAINING SETTINGS                 #
#======================================================#

# Length of an EEG epoch in seconds
EPOCH_LENGTH = 1
# How many letters are used in training
NUM_TRIALS = 5
# How any flashes of all rows and columns, sequence, per trial
SEQ_PER_TRIAL = 3
# Sampling rate of the headset in Hz (2000Hz BioSemi (speed 4)) (250Hz OpenBCI)
SAMPLING_RATE = 250
# What channels are we using in analysis (specify indices)
CHANNELS = [0, 1, 3, 4, 5, 6, 7]
# File path to the pickle file containing the trained classifiers
CLASSIFIER_FILENAME = 'p300_classifier.pkl'
# Number of samples that should be in each epoch
SAMPLES_PER_EPOCH = EPOCH_LENGTH * SAMPLING_RATE
# Sampling rate to down sample data to
DOWN_SAMPLE_TARGET = 128
# Samples in the raw data to skip in order to get desired sample rate
SAMPLES_TO_SKIP = int(math.ceil(float(SAMPLING_RATE) / DOWN_SAMPLE_TARGET))
# Max number of samples an epoch may have after down sampling
MAX_NUM_SAMPLES_POSSIBLE = (SAMPLING_RATE * EPOCH_LENGTH) / SAMPLES_TO_SKIP

#======================================================#
#                    DATA FILTERING                    #
#======================================================#

# Option to use filter
FILTER_DATA = False
# Low cut-off frequency value (Hz)
HIGHPASS_CUTOFF = .1
# High cut-off frequency value (Hz)
LOWPASS_CUTOFF = 30
# Order of the filter
FILTER_ORDER = 6

#======================================================#
#                   GUI SETTINGS                       #
#======================================================#

# Width of character grid in pixels
GRID_WIDTH = 500
# Time in miliseconds the rectangle remains over a given row/column
HIGHLIGHT_TIME = 100
# Time in miliseconds between rectangle presentions
INTERMEDIATE_TIME = 80
# Will the highlighting rectangle move randomly
RANDOM_HIGHLIGHT = True
# Colors of the characters in the grid
DEFAULT_CHAR_COLOR = '#ffffff'
HIGHLIGHT_CHAR_COLOR = '#000000'
# Color for rectangle that highlights a single character
CHAR_SELECT_COLOR = '#00ff00'
# Color of highlighting rectangle
RECT_COLOR = '#ffffff'
# Backgrounf color of the character grid/canvas
GRID_BG_COLOR = '#000000'

#======================================================#
#                DATA OUTPUT SETTINGS                  #
#======================================================#

# Output directory
CSV_DIRECTORY = 'csv'

#======================================================#
#             DATA VISUALIZATION SETTINGS              #
#======================================================#

# When visualizing our epochs do we want to dislay the individual epochs
DISPLAY_TRIALS = False
# When visualizing epochs do we want to display the average
DISPLAY_AVERAGE = True
