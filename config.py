# Default port for accessing the OpenBCI board
PORT = '/dev/ttyUSB0'
# Length of an EEG epoch in seconds
EPOCH_LENGTH = 1
# Number of complete iterations before epochs
# are filed and averaged
ITERS_BETWEEN_ANALYSIS = 3
# Do we want to log OpenBCI output
LOG_OUTPUT = True
# Log file name
LOG_FILENAME = "p300SpellerLog.txt"
# Time in seconds the rectangle remains over a given row/column
HIGHLIGHT_TIME = .100
# Time in milis between rectangle presentions
INTERMEDIATE_TIME = 80
# Save collected Data to CSV files
OUTPUT_CSV = True
# CSV directory
CSV_DIRECTORY = "csv/"
