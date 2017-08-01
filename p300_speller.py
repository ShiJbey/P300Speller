"""
This module runs the p-300 speller application.
"""
# Standard Library
import os
import pickle
from multiprocessing import Process, Queue, Pipe
import datetime
import random
import Tkinter as tk
# External library code
import numpy as np
from sklearn import svm
# Custom modules
from epoch import *
from butter_filters import butter_bandpass_filter, butter_highpass_filter, butter_lowpass_filter
from build_classifier import *
from eeg_event_codes import *
import config

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_DELIM = '/'
if os.name == 'nt':
    PATH_DELIM = '\\'

def get_closest_epoch_time(target_time, epoch_times):
    """
    Returns the closest epoch start time to a given target time
    """
    closest_time = -1
    for ep_time in epoch_times:
        if ep_time[0] <= target_time:
            closest_time = ep_time
            epoch_times.remove(ep_time)
        else:
            break
    return closest_time

def write_raw_to_csv(data_hist, epoch_times, file_path, delim=","):
    """
    Given an array of sample data arrays, writes the data to a file
    """
    print np.shape(data_hist)
    # Sort the epoch times
    epoch_times = sorted(epoch_times)
    with open(str(file_path), 'a+') as out_file:
        for sample in data_hist:
            csv_row = ""
            for index in range(len(sample)):
                if index == 0:
                    # Write the time of the sample
                    sample_time = sample[index]
                    csv_row += str(repr(sample_time))
                    csv_row += delim
                    if len(epoch_times) > 0:
                        closest_time = get_closest_epoch_time(sample_time, epoch_times)
                        if closest_time == -1:
                            csv_row += str(repr(-1))
                            csv_row += delim
                        else:
                            csv_row += str(closest_time[1]) # Write out the event code
                            csv_row += delim
                    else:
                        csv_row += str(repr(-1))
                        csv_row += delim 
                else:
                    channel_voltage = sample[index]
                    csv_row += str(repr(channel_voltage))
                    csv_row += delim
            csv_row = csv_row[:-1]
            csv_row += "\n"
            out_file.write(csv_row)

def get_code_time_pairs(time_code_matrix):
    time_code_pairs = []
    for index in range(len(time_code_matrix)):
        if time_code_matrix[index,1] != 32:
            time_code_pairs.append((time_code_matrix[index,0], time_code_matrix[index,1]))
    return time_code_pairs

def trim_data_history(data_history, start_time, end_time, num_channels=len(config.CHANNELS)):
    """
    Returns a matrix of data exluding any samples that fell
    outside of the time interval
    """
    trimmed_history = np.zeros((0,1 + num_channels), dtype=np.float64)
    
    for row_index in range(len(data_history)):
        sample = data_history[row_index, :]
        sample_time = sample[0] 
        if sample_time >= start_time and sample_time <= end_time:
            trimmed_history = np.vstack((trimmed_history, sample))
    
    return trimmed_history

def output_averaged_epoch_list(avg_list, file_path, name_root, trial_num, p300_index):
    """
    Outputs the data held inside of the row/col_average lists to a .csv file
    """
    for rc_index in range(len(avg_list)):
        rc = avg_list[rc_index]
        filename = file_path + name_root + "_" + str(rc_index) + "_" + str(trial_num) + ".csv"
        with open(filename, "w+") as outfile:
            if rc_index == p300_index:
                outfile.write("1\n")
            else:
                outfile.write("0\n")
            for sample in rc:
                csv_row = ""
                for col in sample:
                    csv_row += str(col)
                    csv_row += ","
                csv_row = csv_row[:-1]
                csv_row += "\n"
                outfile.write(csv_row)

def select_rand_element(x):
    """
    Selects element from a given list and returns it
    """
    rand_index = random.randint(0, len(x) - 1)
    elem = x[rand_index]
    return elem

def get_desired_channels(sample, index_list):
    """
    Returns python list of desired channel data from an EEG sample
    """
    new_sample = []
    for i in range(len(sample)):
        if i in index_list:
            new_sample.append(sample[i])
    return new_sample


def get_p300_prediction(clf, average_data, sampling_rate=config.SAMPLING_RATE):
    """
    Makes a prediction on a row/column index given a classifier and list of sample data matrices
    """
    best_prediction = {"index": 0, "confidence": 0.0}

    for index in range(len(average_data)):
        data = down_sample_data(average_data[index], sampling_rate, target_sample_rate=128)
        data = np.ravel(data)
        predicted_class = clf.predict(np.reshape(data,(1, -1)))
        #prediction_confidence = clf.predict_proba(np.reshape(data,(1, -1)))[0][1]
        prediction_confidence = clf.decision_function(np.reshape(data,(1, -1)))[0]
        print "Row/Col: %d" % index
        print "Class: "
        print predicted_class
        print "Confidence:"
        print prediction_confidence
        if predicted_class[0] == '1' and (prediction_confidence > best_prediction["confidence"]):
            best_prediction["index"] = int(index)
            best_prediction["confidence"] = prediction_confidence
    return best_prediction["index"]

if __name__ == '__main__':
    import speller_gui as gui
    from pylsl import StreamInlet, resolve_stream
    import argparse

    #==========================================================#
    #             Collect command line arguments               #
    #==========================================================#
    
    parser = argparse.ArgumentParser(description='Runs the P300 application')
    parser.add_argument('-t','--train',
                        dest='training_mode',
                        action='store_true',
                        default=True,
                        help='Runs the gui in training mode and '
                        'creates a new classifier with the data.')
    parser.add_argument('-l','--live',
                        dest='live_mode',
                        action='store_true',
                        default=False,
                        help='Runs the gui in live mode and '
                        'loads a classifier for making spelling predictions.')
    parser.add_argument('-s', '--simulate',
                        dest="simulation_data_path",
                        type=str,
                        nargs=1,
                        default="",
                        help='Runs the speller using simulation data')
    parser.add_argument('-c', '--classifier',
                        dest='clf_path',
                        type=str,
                        nargs=1,
                        default=None,
                        help='Specifies a path for importing a classifier .pkl file')
    parser.add_argument('--gui-only',
                        dest="gui_only",
                        action='store_true',
                        default=False,
                        help='Only runs the gui in the live mode '
                        'without connecting to LSL data stream.')
    parser.add_argument('--output-raw',
                        dest="output_raw",
                        action='store_true',
                        default=True,
                        help='Writes the raw data collected to a'
                        'CSV file to be used at a later time.')
    parser.add_argument('--output-epochs',
                        dest="output_epochs",
                        action='store_true',
                        default=False,
                        help='Writes the epoch data collected to'
                        ' CSV files.')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help = 'Prints the current status of the'
                        'application to the terminal')
    args = parser.parse_args()

    if args.live_mode:
        args.training_mode = False

    if (args.live_mode and args.training_mode or
        args.training_mode and args.simulation_data_path != '' or
        args.live_mode and args.simulation_data_path != ''):
        raise parser.error("Only one mode (train, live, or simulate) may be specified at a time.")

    #==========================================================#
    #                    Import Classifier                     #
    #==========================================================#

    classifier = None
    # Exit application if no classifier '.pkl' file can be found
    if args.live_mode:
        clf_import_path = ""
        if args.clf_path:
            clf_import_path = args.clf_path[0]
        else:
            clf_import_path = DIR_PATH + PATH_DELIM + str(config.CLASSIFIER_FILENAME)

        if os.path.exists(clf_import_path):
            
            # Import the classifier
            if args.verbose:
                print "Attempting to import classifier from pickle file..."

            pkl_file = open(clf_import_path, 'rb')
            
            classifier = pickle.load(pkl_file)

            pkl_file.close()

            if args.verbose:
                print "Classifier loaded from file."
        else:
            raise IOError("No classifier file found at: " + clf_import_path)

    #==========================================================#
    #              Set-up for Data Exporting                   #
    #==========================================================#
    
    # Sets up path for outputing the raw data    
    OUTPUT_DIR = DIR_PATH + PATH_DELIM + str(config.CSV_DIRECTORY) + PATH_DELIM + datetime.datetime.now().strftime("%B_%d_%Y@%I-%M-%S%p") + PATH_DELIM
    RAW_DATA_FILENAME = datetime.datetime.now().strftime("%B_%d_%Y@%I-%M-%S%p") + "RawData.csv"
    
    if not os.path.exists(OUTPUT_DIR):
        if args.verbose:
            print "Output directory does not exist"
            print "Creating output directory at: %s" % (OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    #==========================================================#
    #                  Set-up Data Structures                  #
    #==========================================================#

    # Queues for epochs that have been created by the GUI
    row_epoch_queue = Queue()
    col_epoch_queue = Queue()

    # Pipe to send a predicted character to the gui to update the spelling buffer
    main_conn, gui_conn = Pipe()

    # 2D numpy array of sample data [[read_time,c1,c2,...cn],...]
    data_history = np.zeros((0,1 + len(config.CHANNELS)), dtype=np.float64)

    # List of times of all the epochs in the current trial
    epoch_times = []

    # Lists of Epochs that have been collected over a trial
    row_epochs = [[], [], [], [], [], []]
    col_epochs = [[], [], [], [], [], []]

    # List of numpy matrices. Each matrix hold the average of the sample data for each channel
    row_averages = []
    col_averages = []

    
    #==========================================================#
    #                    Spawn GUI Process                     #
    #==========================================================#
    
    # Create processes for the GUI and the EEG collection
    gui_process = Process(target=gui.run_gui, args=(row_epoch_queue, col_epoch_queue, gui_conn, args.training_mode))
    # Start the GUI process
    gui_process.start()

    #==========================================================#
    #        Connect to LSL Stream or Simulation Data          #
    #==========================================================#

    if not args.gui_only and (args.live_mode or args.training_mode):
        # Look for the stream of EEG data on the network
        if args.verbose:
            print "Looking for EEG stream..."
        streams = resolve_stream('type', 'EEG')
        if args.verbose:
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
    epoch = None               # Reference to the most recent epoch
    first_epoch = None         # Reference to the first epoch created in the current trial
    last_epoch = None          # Reference to the last epoch created in the current trial

    if args.training_mode:
        # Select a random row and column for training this trial
        p300_row = select_rand_element(row_options)
        p300_col = select_rand_element(col_options)
        # Send these over the pipe to the GUI
        main_conn.send(["train", "row", p300_row])
        main_conn.send(["train", "col", p300_col])

    # Run data collection for as long as the gui is active
    while gui_process.is_alive():
        
        #==========================================================#
        #            Read Epochs Created by the GUI                #
        #==========================================================#

        # Try reading from both queues and updating the data dictionaries
        try:
            if row_epochs_read < 6:
                epoch = row_epoch_queue.get_nowait()
                if epoch:
                    epoch_times.append((epoch.start_time, epoch.get_event_code()))
                    row_epochs[epoch.index].append(epoch)
                    row_epochs_read += 1
                    if first_epoch == None or epoch.start_time < first_epoch.start_time:
                        first_epoch = epoch
                    if last_epoch == None or epoch.start_time > last_epoch.start_time:
                        last_epoch = epoch

            if col_epochs_read < 6:
                epoch = col_epoch_queue.get_nowait()
                if epoch:
                    epoch_times.append((epoch.start_time, epoch.get_event_code()))
                    col_epochs[epoch.index].append(epoch)
                    col_epochs_read += 1
                    if first_epoch == None or epoch.start_time < first_epoch.start_time:
                        first_epoch = epoch
                    if last_epoch == None or epoch.start_time > last_epoch.start_time:
                        last_epoch = epoch
        except:
            pass

        #==========================================================#
        #                 Read Sample From LSL                     #
        #==========================================================#

        if not args.gui_only and (args.live_mode or args.training_mode):
            # Get sample and time_stamp from data stream
            sample, time_stamp = inlet.pull_sample(timeout=0.0)
            # Store the sample in the data history
            if sample != None and first_epoch != None and time_stamp >= first_epoch.start_time and time_stamp <= last_epoch.start_time + config.EPOCH_LENGTH:
                sample_row = np.append(time_stamp, get_desired_channels(sample, config.CHANNELS))     
                data_history = np.vstack((data_history, sample_row))
        

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
            if args.verbose:
                print "Trial over!"
            #==========================================================#
            #                Obtaining Remaining Data                  #
            #==========================================================#
            
            if not args.gui_only:

                if args.live_mode or args.training_mode:
                    # Get remaining data pertaining to last epoch

                    if args.verbose:
                        print "Getting remaining data from queue..."

                    while time_stamp <= last_epoch.start_time + config.EPOCH_LENGTH + .5:
                        
                        sample, time_stamp = inlet.pull_sample(timeout=0.1)
                        if sample != None:
                            sample_row = np.append(time_stamp, get_desired_channels(sample, config.CHANNELS))
                            if np.shape(data_history)[0] == 0:
                                data_history = np.vstack((data_history, sample_row))
                            else:
                                data_history = np.vstack((data_history, sample_row))
                        else:
                            break                  

                    if args.output_raw:
                        # Write the contents of the
                        write_raw_to_csv(data_history, epoch_times, OUTPUT_DIR + RAW_DATA_FILENAME)
                        if args.verbose:
                                print "Wrote raw data to \'%s\' in the output directory." % (RAW_DATA_FILENAME)
                        
                    # Erase any samples that fall outside of this sequence
                    data_history = trim_data_history(data_history, first_epoch.start_time, last_epoch.start_time + config.EPOCH_LENGTH + .5)
            
            #==========================================================#
            #                     Filtering Data                       #
            #==========================================================#

            if config.FILTER_DATA:
                if args.verbose:
                    print "Filtering Data"

                # Replace each column in data_history with the filtered version
                if len(data_history) > 0:
                    for column_index in range(len(data_history[0])):
                        if column_index > 0:                            
                            data_history[:,column_index] = butter_highpass_filter(data_history[:,column_index],
                                                                                config.HIGHPASS_CUTOFF,
                                                                                config.SAMPLING_RATE,
                                                                                order=config.FILTER_ORDER)
                            
                            data_history[:,column_index] = butter_lowpass_filter(data_history[:,column_index],
                                                                                config.LOWPASS_CUTOFF,
                                                                                config.SAMPLING_RATE,
                                                                                order=config.FILTER_ORDER)
                            
            #==========================================================#
            #               Splitting Data into Epochs                 #
            #==========================================================#

            # Fill and move epochs
            if args.verbose:
                print "Splitting into epochs."
            for row in row_epochs:
                for ep in row:
                    ep.get_epoch_data(data_history, config.SAMPLES_PER_EPOCH)
            for col in col_epochs:
                for ep in col:
                    ep.get_epoch_data(data_history, config.SAMPLES_PER_EPOCH)
            if args.verbose:
                print "Done splitting."

            # Remove any epochs that do not have the desired samples
            if args.verbose:
                print "Rejecting any epochs that dont have the desired number of samples"
            row_epochs = reject_epochs_from_matrix(row_epochs, samples_per_epoch=config.SAMPLES_PER_EPOCH)
            col_epochs = reject_epochs_from_matrix(col_epochs, samples_per_epoch=config.SAMPLES_PER_EPOCH)
            

            # Clear all sample data up to the start_time of last epoch
            data_history = np.zeros((0,1 + len(config.CHANNELS)), dtype=np.float64)

            # Clear all epoch times
            epoch_times = []
            if args.verbose:
                print "Data history has been cleared for next trial."

            #==========================================================#
            #                Averaging Together Epochs                 #
            #==========================================================#

            if not args.gui_only:

                if args.verbose:
                    print "Averaging the epochs for each row/column"
                row_averages = average_epoch_matrix(row_epochs, config.SAMPLES_PER_EPOCH, len(config.CHANNELS))
                col_averages = average_epoch_matrix(col_epochs, config.SAMPLES_PER_EPOCH, len(config.CHANNELS))

                if args.training_mode:
                    if args.verbose:
                        print "Outputing epoch averages for this trial"
                    output_averaged_epoch_list(row_averages, OUTPUT_DIR, "Col", trials_complete, p300_col)
                    output_averaged_epoch_list(col_averages, OUTPUT_DIR, "Row", trials_complete, p300_row)
                
                if args.output_epochs:
                    if args.verbose:
                        print "Outputing regular epoch for this trial"
                    output_epoch_matrix(row_epochs, directory=OUTPUT_DIR)
                    output_epoch_matrix(col_epochs, directory=OUTPUT_DIR)
            
                if args.verbose:
                    print "Clearing epoch data"
                row_epochs = [[], [], [], [], [], []]
                col_epochs = [[], [], [], [], [], []]

            #==========================================================#
            #                     Classification                       #
            #==========================================================#
            if not args.gui_only and args.live_mode:

                if args.verbose:
                    print "Making prediction..."

                predicted_row = get_p300_prediction(classifier, row_averages)
                predicted_col = get_p300_prediction(classifier, col_averages)

                if args.verbose:
                    print "Classifier predicted Row: %d and Col: %d" % (predicted_row, predicted_col)
                    
                main_conn.send(["prediction", "row", predicted_row])
                main_conn.send(["prediction", "col", predicted_col])

            #==========================================================#
            #                 Prep for the Next Trial                  #
            #==========================================================#         

            # Reset averages to all zeros
            for i in range(6):
                row_averages[i] = np.zeros((config.SAMPLES_PER_EPOCH,len(config.CHANNELS)))
                col_averages[i] = np.zeros((config.SAMPLES_PER_EPOCH,len(config.CHANNELS)))
            
            # Reset counter
            sequences_complete = 0
            epoch = None
            first_epoch = None
            last_epoch = None

            if args.training_mode:
                # Select a random row and column for training this trial
                p300_row = select_rand_element(row_options)
                p300_col = select_rand_element(col_options)
                # Send these over the pipe to the GUI
                main_conn.send(["train", "row", p300_row])
                main_conn.send(["train", "col", p300_col])
                # Clear data history for the next trial
                data_history = np.zeros((0,1 + len(config.CHANNELS)), dtype=np.float64)
    
    #==========================================================#
    #                 Disconnect LSL Stream                    #
    #==========================================================#

    if not args.gui_only and (args.live_mode or args.training_mode):
        inlet.close_stream()
        if args.verbose:
            print "Closed data stream."