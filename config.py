#======================================================#
#             DATA COLLECTION SETTINGS                 #
#======================================================#

# Length of an EEG epoch in seconds
EPOCH_LENGTH = 1
# Number of complete iterations before epochs
# are filed and averaged
ITERS_BETWEEN_ANALYSIS = 3
# Sampling rate of the headset (2kHz BioSemi (speed 4)) (250Hz OpenBCI)
SAMPLING_RATE = 2000
# How many channels are we using
NUM_CHANNELS = 4



#======================================================#
#                   GUI SETTINGS                       #
#======================================================#

# Width and height of the window
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
# Time in miliseconds the rectangle remains over a given row/column
HIGHLIGHT_TIME = 100
# Time in miliseconds between rectangle presentions
INTERMEDIATE_TIME = 100
# Will the highlighting rectangle move randomly
RANDOM_HIGHLIGHT = True

#======================================================#
#              DATA ANALYSIS SETTINGS                  #
#======================================================#

# Save collected Data to CSV files
OUTPUT_CSV = True
# CSV directory
CSV_DIRECTORY = "csv"
# When visualizing our epochs do we want to dislay the individual epochs
DISPLAY_TRIALS = False
# When visualizing epochs do we want to display the average
DISPLAY_AVERAGE = True


