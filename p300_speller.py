"""
This module runs the p-300 speller using multiple processes.
Processes are spawned for the main module, the eeg data
collection, and the GUI. The eeg_process is responsible for
sending sample data to the data buffer managed by the main module.
The gui process is responsible for updating the gui and recording
times when the gui has been updated
"""
# Standard Library
import os
import argparse
import pickle
from multiprocessing import Process, Queue, Pipe
import datetime
import random
import Tkinter as tk
# External library code
from pylsl import StreamInlet, resolve_stream
import numpy as np
from sklearn import svm
# Custom modules
import butterworth_bandpass
import speller_gui as gui
import config

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_DELIM = '/'
if os.name == 'nt':
    PATH_DELIM = '\\'

class EEGEpoch(object):
    """Manages segments of EEG epoch data"""
    def __init__(self, is_row, index, start_time, is_p300=False):
        self.is_row = is_row            # Boolean indicating if this epoch is for a row or column
        self.index = index              # Row/Col index of this epoch
        self.start_time = start_time    # Starting time of this epoch
        self.sample_data = []           # 2D array of samples of channel data
        self.is_p300 = is_p300          # Set to true if there is supposed to be P300 (used in training) 

    def is_within_epoch(self, sample_time):
        """Returns true if the given time is within this given epoch"""
        return (sample_time >= self.start_time
                and sample_time <= self.start_time + config.EPOCH_LENGTH)

    def __str__(self):
        return ("Start: %s \nActive?: %s \nNum Samples: %s"
                %(self.start_time, self.active, len(self.sample_data)))

    def output_to_csv(self, epoch_num, directory=str(config.CSV_DIRECTORY),delim=","):
        """Output the data held by this epoch to a csv file"""
        filename = ""
        if self.is_row:
            filename += "Row_"
        else:
            filename += "Col_"
        filename += str(self.index) + "_epoch" + "_" + str(epoch_num) + ".csv"

        # First line has the starting time
        with open(directory + str(filename), 'w') as out_file:
            out_file.write(str(repr(self.start_time)) + "\n")
            # All following lines are organized one sample per
            #  row and each column is a different channel
            for sample in self.sample_data:
                csv_row = ""
                for voltage in sample:
                    csv_row += str(voltage)
                    csv_row += delim
                csv_row = csv_row[:-1]
                csv_row += "\n"
                out_file.write(csv_row)

def get_epoch_data(start_time, data_hist):
    """Loops through the data history and returns a 2D array of sample data"""
    # 2D array that will hold channel data from samples
    data = []

    # Loop through the list of keys until we find the sample
    #  closest to the start time
    closest_time = -1
    for read_time in sorted(data_hist.keys()):
        if read_time <= start_time:
            # Overwite until we get a time closes to the start_time
            closest_time = read_time
        elif read_time > start_time:
            break

    # Return empty list
    if closest_time == -1:
        return data

    # Get all samples for the epic
    start_sample_index = sorted(data_hist.keys()).index(closest_time)
    samples_per_epoch = config.EPOCH_LENGTH * config.SAMPLING_RATE
    sample_times = sorted(data_hist.keys())[
        start_sample_index:start_sample_index + (samples_per_epoch + 1)]

    # Append the sample data for all the samples
    for time in sample_times:
        data.append(data_hist[time])
    return data

def update_data_history(history, time_stamp, sample):
    """Adds the sample channel data to the dict under its read time"""
    history[time_stamp] = sample

def select_rand_element(x):
        """Selects element from list 'x' and returns it"""
        rand_index = random.randint(0, len(x) - 1)
        elem = x[rand_index]
        return elem
    
def run_gui(row_epoch_queue, col_epoch_queue, pipe_conn, is_training):
    """Starts the p300 speller gui"""
    root = tk.Tk()
    root.title("P300 Speller")
    #root.geometry('%sx%s'%(config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    root.protocol("WM_DELETE_WINDOW", root.quit)
    speller_gui = gui.P300GUI(root,
                            row_epoch_queue,
                            col_epoch_queue,
                            pipe_conn,
                            is_training)
    start_screen = gui.StartScreen(root, speller_gui)
    start_screen.display_screen()
    root.mainloop()



if __name__ == '__main__':
    #==========================================================#
    #             Collect command line arguments               #
    #==========================================================#

    def str2bool(val):
        """Converts a string value to a boolean value."""
        if val.lower() in ('yes', 'true', 't', 'y', '1', ''):
            return True
        elif val.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    
    parser = argparse.ArgumentParser(description='Runs the P300 application')
    parser.add_argument('--mode',
                        dest='mode',
                        type=str,
                        metavar='mode',
                        default='train',
                        help='Runs the gui in training mode and '
                        'creates a new classifiers with the data')
    parser.add_argument('--gui_only',
                        metavar='gui_only',
                        type=str2bool,
                        default=False,
                        help='Only runs the gui in the live mode '
                        'without connecting to LSL data stream')
    args = parser.parse_args()

    if not (args.mode == 'train' or args.mode == 'live'):
        raise argparse.ArgumentTypeError('Mode needs tp be either \'train\' or \'live\'')

    #==========================================================#
    #              Import and Set-up Classifier                #
    #==========================================================#

    classifier = None
    # Exit application if no classifier '.pkl' file can be found
    if args.mode == 'live':
        if os.path.exists(DIR_PATH + PATH_DELIM + str(config.CLASSIFIER_FILENAME)):
            # Import the classifier
            pkl_file = open(DIR_PATH + PATH_DELIM + str(config.CLASSIFIER_FILENAME), 'rb')
            classifier = pickle.load(pkl_file)
        else:
            raise IOError("No classifier file found at: " + DIR_PATH + PATH_DELIM + str(config.CLASSIFIER_FILENAME))

    #==========================================================#
    #                  Set-up Data Structures                  #
    #==========================================================#

    # Queues for epochs that have been created by the GUI
    row_epoch_queue = Queue()
    col_epoch_queue = Queue()

    # Pipe to send a predicted character to the gui to update the spelling buffer
    main_conn, gui_conn = Pipe()

    # Dictionary of data samples collected in this sequence (time_stamp -> channel_data)
    data_history = {}

    # List of dictionaries holding epoch objects
    row_data = range(6)
    row_data[0] = {"active":[], "past":[]}
    row_data[1] = {"active":[], "past":[]}
    row_data[2] = {"active":[], "past":[]}
    row_data[3] = {"active":[], "past":[]}
    row_data[4] = {"active":[], "past":[]}
    row_data[5] = {"active":[], "past":[]}
    col_data = range(6)
    col_data[0] = {"active":[], "past":[]}
    col_data[1] = {"active":[], "past":[]}
    col_data[2] = {"active":[], "past":[]}
    col_data[3] = {"active":[], "past":[]}
    col_data[4] = {"active":[], "past":[]}
    col_data[5] = {"active":[], "past":[]}

    # List of numpy arrays holding sample data for a single trial
    samples_per_epoch = config.EPOCH_LENGTH * config.SAMPLING_RATE
    row_averages = range(6)
    row_averages[0] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    row_averages[1] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    row_averages[2] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    row_averages[3] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    row_averages[4] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    row_averages[5] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages = range(6)
    col_averages[0] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages[1] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages[2] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages[3] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages[4] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
    col_averages[5] = np.zeros((samples_per_epoch,len(config.CHANNELS)))

    # Holds all of the training examples
    X = []
    # Holds all of the example classifications
    y = []
    
    #==========================================================#
    #                    Spawn GUI Process                     #
    #==========================================================#
    
    # Create processes for the GUI and the EEG collection
    gui_process = Process(target=run_gui, args=(row_epoch_queue, col_epoch_queue, gui_conn, (args.mode == 'train')))
    # Start the GUI process
    gui_process.start()

    #==========================================================#
    #                 Connect to LSL Stream                    #
    #==========================================================#

    if not args.gui_only:
        # Look for the stream of EEG data on the network
        print "Looking for EEG stream..."
        streams = resolve_stream('type', 'EEG')
        print "Stream found!"
        # Create inlet to read from stream
        inlet = StreamInlet(streams[0])
    
    #==========================================================#
    #             Data Collection and Manipulation             #
    #==========================================================#

    trials_complete = 0        # Number of completed trials
    sequences_complete = 0     # Number of completed sequences (each row/column flashes once)
    row_epochs_read = 0        # How many row epochs read this sequence
    col_epochs_read = 0        # How many column epochs read this sequence
    row_options = range(6)     # List of row indices for selecting random characters
    col_options = range(6)     # List of column indices for selecting random characters
    p300_row = -1              # Row that has been randomly selected for training
    p300_col = -1              # Column that has been randomly selected for training

    if args.mode == 'train':
        # Select a random row and column for training this trial
        p300_row = select_rand_element(row_options)
        p300_col = select_rand_element(col_options)
        # Send these over the pipe to the GUI
        main_conn.send(["train", "row", p300_row])
        main_conn.send(["train", "col", p300_col])

    # Run data collection for as long as the gui is active
    while gui_process.is_alive():

        #==========================================================#
        #                 Read Sample From LSL                     #
        #==========================================================#

        if not args.gui_only:
            # Get sample and time_stamp from data stream
            sample, time_stamp = inlet.pull_sample(timeout=0.0)
            # Store the sample locally
            if sample != None:
                update_data_history(data_history, time_stamp, sample)

        #==========================================================#
        #            Read Epochs Created by the GUI                #
        #==========================================================#

        # reference to the most recent epoch
        epoch = None
        
        # Try reading from both queues and updating the data dictionaries
        try:
            if row_epochs_read < 6:
                epoch = row_epoch_queue.get_nowait()
                if epoch:
                    row_data[epoch.index]["active"].append(epoch)
                    row_epochs_read += 1
            if col_epochs_read < 6:
                epoch = col_epoch_queue.get_nowait()
                if epoch:
                    col_data[epoch.index]["active"].append(epoch)
                    col_epochs_read += 1
        except:
            pass

        #==========================================================#
        #          Processing the End of the Sequence              #
        #==========================================================#
        
        # Sequence complete
        if col_epochs_read == 6 and row_epochs_read == 6:
            sequences_complete += 1
            col_epochs_read = 0
            row_epochs_read = 0

        #==========================================================#
        #            Processing the End of the Trial               #
        #==========================================================#

        # Trial complete
        if sequences_complete >= config.SEQ_PER_TRIAL:
            trials_complete += 1

            # Get remaining data pertaining to last epoch
            if not args.gui_only:
                while time_stamp != None and time_stamp <= epoch.start_time + config.EPOCH_LENGTH:
                    print "Getting remaining data from queue..."
                    sample, time_stamp = inlet.pull_sample(timeout=0.1)
                    # Extract only the data from the channels that we want
                    sample_data = []
                    for channel in range(len(config.CHANNELS)):
                        sample_data.append(sample[channel])
                    update_data_history(time_stamp, sample_data)

            #==========================================================#
            #                     Filtering Data                       #
            #==========================================================#

            # Move all data to an np array
            all_data = []
            for sample in data_history:
                all_data.append[sample]
            all_data = np.array(all_data)

            # Replace each column with the filtered version
            if len(all_data) > 0:
                for channel in range(len(all_data[0])):
                    filtered_data = butter_bandpass_filter(all_data[:,channel],
                                                            config.FILTER_LOWCUT,
                                                            config.FILTER_HIGHCUT,
                                                            config.SAMPLING_RATE,
                                                            order=config.FILTER_ORDER)
                    all_data[:,channel] = filtered_data

            # Update the data history dictionary
            for i in range(len(all_data)):
                data_hist.keys[i] = all_data[i,:].tolist()

            #==========================================================#
            #               Splitting Data into Epochs                 #
            #==========================================================#
            
            # Fill and move epochs
            print "Splitting into epochs."
            for row in range(len(row_data)):
                for ep in row_data[row]["active"]:
                    ep.sample_data = get_epoch_data(ep.start_time, data_history)
                    row_data[row]["past"].append(ep)
                    row_data[row]["active"].remove(ep)
            for col in range(len(col_data)):
                for ep in col_data[col]["active"]:
                    ep.sample_data = get_epoch_data(ep.start_time, data_history)
                    col_data[col]["past"].append(ep)
                    col_data[col]["active"].remove(ep)
            print "Done splitting."

            # Clear all data up to the start_time of last epoch
            for key in sorted(data_history.keys()):
                if key <= epoch.start_time:
                    del data_history[key]

            #==========================================================#
            #                Averaging Together Epochs                 #
            #==========================================================#

            # Average the rows
            for row in range(len(row_data)):
                # Sum the channel values
                for ep in row_data[row]["past"]:
                    data = np.array(ep.sample_data);
                    for sample_num in range(len(data)):
                        row_averages[row][sample_num,0:config.NUM_CHANNELS - 1] = row_averages[row][sample_num,0:config.NUM_CHANNELS - 1] + data[sample_num,0:config.NUM_CHANNELS - 1]
                # Divide by the # of epochs for the row
                row_averages[row] = row_averages[row] / len(row_data[row]["past"])

            # Average the columns
            for col in range(len(col_data)):
                # Sum the channel values
                for ep in col_data[col]["past"]:
                    data = np.array(ep.sample_data);
                    for sample_num in range(len(data)):
                        col_averages[col][sample_num,0:config.NUM_CHANNELS - 1] = col_averages[col][sample_num,0:config.NUM_CHANNELS - 1] + data[sample_num,0:config.NUM_CHANNELS - 1]
                # Divide by the # of epochs for the column
                col_averages[col] = col_averages[col] / len(col_data[col]["past"])

            #==========================================================#
            #         Classification or Storage for Training           #
            #==========================================================#
            if args.mode == 'live':
                predicted_col = (-1, -1)
                predicted_row = (-1, -1)

                # pass row data from this past trial to the trained classifier
                for row_index in range(len(row_averages)):
                    prediction = classifier.predict(np.transpose(row_averages[row_index]))
                    false_count = prediction.count(0)
                    pos_count = prediction.count(1)
                    if pos_count >= false_count and pos_count > predicted_row[1]:
                        predicted_row = (row_index, pos_count)
                # pass col data from this past trial to the trained classifier
                for col_index in range(len(col_averages)):
                    prediction = classifier.predict(np.transpose(col_averages[col_index]))
                    false_count = prediction.count(0)
                    pos_count = prediction.count(1)
                    if pos_count >= false_count and pos_count > predicted_col[1]:
                        predicted_col = (col_index, pos_count)

                main_conn.send(["prediction", "row", predicted_row])
                main_conn.send(["prediction", "col", predicted_col])

            else:
                # Add all row average data to the example lists
                for row_index in range(len(row_averages)):
                    for channel_index in range(len(row_averages[row_index][0])):
                        X.append(row_averages[row_index][:channel_index].tolist())
                        if row_index == p300_row:
                            y.append(1)
                        else:
                            y.append(0)
                # Add all col average data to the example lists
                for col_index in range(len(col_averages)):
                    for channel_index in range(len(col_averages[col_index][0])):
                        X.append(col_averages[col_index][:channel_index].tolist())
                        if col_index == p300_col:
                            y.append(1)
                        else:
                            y.append(0)

                

            # Reset averagees to all zeros
            for i in range(6):
                row_averages[i] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
                col_averages[i] = np.zeros((samples_per_epoch,len(config.CHANNELS)))
            
            # Reset counter
            sequences_complete = 0

            if args.mode == 'train':
                # Select a random row and column for training this trial
                p300_row = select_rand_element(row_options)
                p300_col = select_rand_element(col_options)
                # Send these over the pipe to the GUI
                main_conn.send(["train", "row", p300_row])
                main_conn.send(["train", "col", p300_col])


    #==========================================================#
    #                       Training                           #
    #==========================================================#

    if args.mode == 'train' and not args.gui_only: 
        classifier = svm.SVC()
        print "Training classifier."
        classifier.fit(X,y)
        # Export the classifier
        print "Exporting classsifier."
        pkl_file = open(DIR_PATH + PATH_DELIM + str(config.CLASSIFIER_FILENAME), 'wb')
        pickle.dump(classifier, pkl_file)
        print "Classifier trained and exported."

    #==========================================================#
    #                         Other                            #
    #==========================================================#

    # Output the the epics to CSV files
    if config.OUTPUT_CSV and not args.gui_only:
        output_dir = DIR_PATH + PATH_DELIM + str(config.CSV_DIRECTORY) + PATH_DELIM + datetime.datetime.now().strftime("%I-%M-%S%p%B_%d_%Y") + PATH_DELIM
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        print "Writing output to: " + output_dir
        for col in range(len(col_data)):
            for i  in range(len(col_data[col]["past"]) - 1):
                ep = col_data[col]["past"][i]
                ep.output_to_csv(i, directory=output_dir)

        for row in range(len(row_data)):
            for i  in range(len(row_data[row]["past"]) - 1):
                ep = row_data[row]["past"][i]
                ep.output_to_csv(i, directory=output_dir)
        print "Done."
