"""
Constants used in running the p300_speller module
or any of its associated modules
"""
#======================================================#
#               DATA/TRAINING SETTINGS                 #
#======================================================#

# Length of an EEG epoch in seconds
EPOCH_LENGTH = 1
# How many letters are used in training
NUM_TRIALS = 36
# How any flashes of all rows and columns, sequence, per trial
SEQ_PER_TRIAL = 7
# Sampling rate of the headset in Hz (2000Hz BioSemi (speed 4)) (250Hz OpenBCI)
SAMPLING_RATE = 250
# What channels are we using in analysis (specify indices)
EEG_CHANNELS = ['Cz', 'P3', 'Pz', 'P4', 'POz']
EEG_TYPES = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg']
CHANNELS_FOR_SVC = ['Pz']
# File path to the pickle file containing the trained classifiers
CLASSIFIER_FILENAME = 'p300_classifier.pkl'
# Down sample factor
DOWN_SAMPLE_FACTOR = 1

#======================================================#
#                    DATA FILTERING                    #
#======================================================#

# Option to use filter
FILTER_DATA = True
# Low cut-off frequency value (Hz)
HIGHPASS_CUTOFF = .1
# High cut-off frequency value (Hz)
LOWPASS_CUTOFF = 60

#======================================================#
#                   GUI SETTINGS                       #
#======================================================#

# Width of character grid in pixels
GRID_WIDTH = 500
# Time in miliseconds the rectangle remains over a given row/column
HIGHLIGHT_TIME = 100
# Time in miliseconds between rectangle presentions
INTERMEDIATE_TIME = 75
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
