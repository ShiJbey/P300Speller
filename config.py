#======================================================#
#               DATA/TRAINING SETTINGS                 #
#======================================================#

# Length of an EEG epoch in seconds
EPOCH_LENGTH = 1
# How many letters are used in training
NUMBER_OF_TRIALS = 10
# How any flashes of all rows and columns, sequence, per trial
SEQ_PER_TRIAL = 12
# Sampling rate of the headset in Hz (2000Hz BioSemi (speed 4)) (250Hz OpenBCI)
SAMPLING_RATE = 2000
# What channels are we using in analysis (specify indices)
CHANNELS = [5,6]
# File path to the pickle file containing the trained classifiers
CLASSIFIER_FILENAME = "p300_classifier.pkl"

#======================================================#
#            DATA FILTERING (BAND PASS)                #
#======================================================#

# Low cut-off frequency value (Hz)
FILTER_LOWCUTT = 0.5
# High cut-off frequency value (Hz)
FILTER_HIGHCUT = 60
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
DEFAULT_CHAR_COLOR = "#ffffff"
HIGHLIGHT_CHAR_COLOR = "#000000"
# Color of highlighting rectangle
RECT_COLOR = "#ffffff"
# Backgrounf color of the character grid/canvas
GRID_BG_COLOR = "#000000"

#======================================================#
#             DATA VISUALIZATION SETTING               #
#======================================================#

# Save collected Data to CSV files
OUTPUT_CSV = False
# CSV directory
CSV_DIRECTORY = "csv"
# When visualizing our epochs do we want to dislay the individual epochs
DISPLAY_TRIALS = False
# When visualizing epochs do we want to display the average
DISPLAY_AVERAGE = True
