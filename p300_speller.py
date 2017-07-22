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
import pylsl
import numpy as np
from sklearn import svm
# Custom modules
from butter_filters import butter_bandpass_filter, butter_highpass_filter, butter_lowpass_filter
from plot_classifier import *
import speller_gui as gui
import config

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_DELIM = '/'
if os.name == 'nt':
    PATH_DELIM = '\\'

SAMPLES_PER_EPOCH = config.EPOCH_LENGTH * config.SAMPLING_RATE

class Epoch(object):

    """Manages segments of EEG epoch data"""
    def __init__(self, is_row, index, start_time, is_p300=False):
        self.is_row = is_row            # Boolean indicating if this epoch is for a row or column
        self.index = index              # Row/Col index of this epoch
        self.start_time = start_time    # Starting time of this epoch
        self.sample_data = None         # Numpy 2D array of sample data (sample X channel)
        self.is_p300 = is_p300          # Set to true if there is supposed to be P300 (used in training)

    def is_within_epoch(self, sample_time):
        """Returns true if the given time is within this given epoch"""
        return (sample_time >= self.start_time
                and sample_time <= self.start_time + config.EPOCH_LENGTH)

    def output_to_csv(self, epoch_num, directory=str(config.CSV_DIRECTORY), delim=","):
        """Output the data held by this epoch to a csv file"""
        filename = ""
        if self.is_row:
            filename += "Row_"
        else:
            filename += "Col_"
        filename += str(self.index) + "_epoch" + "_" + str(epoch_num) + ".csv"

        # First line has the starting time
        with open(directory + str(filename), 'w+') as out_file:
            out_file.write(str(repr(self.start_time)))
            # Write a 0 or 1 indicating classification
            if self.is_p300:
                out_file.write(", 1\n")
            else:
                out_file.write(", 0\n")
            
            # All following lines are organized one sample per
            # row and each column is a different channel
            for sample in self.sample_data:
                csv_row = ""
                for voltage in sample:
                    csv_row += str(voltage)
                    csv_row += delim
                csv_row = csv_row[:-1]
                csv_row += "\n"
                out_file.write(csv_row)

    def get_epoch_data(self, data_hist):
        """Gets all the data for this epoch from a given array of sample data history"""
        index_of_first_sample = -1

        for sample_index in range(len(data_hist)):
            if data_hist[sample_index,0] <= self.start_time:
                index_of_first_sample = sample_index
            elif data_hist[sample_index,0] > self.start_time:
                if index_of_first_sample == -1:
                    index_of_first_sample = sample_index
                    break
        
        self.sample_data = np.array(data_hist[
                                        index_of_first_sample:index_of_first_sample + SAMPLES_PER_EPOCH + 1,1:],
                                    dtype=np.float64)

def write_raw_to_csv(data_hist, reference_time, file_path, delim=","):
    """Given an array of sample data arrays, writes them to a file"""
    with open(str(file_path), 'a+') as out_file:
        for sample in data_hist:
            csv_row = ""
            for col_index in range(len(sample)):
                if col_index == 0:
                    # Writes the time of this sample as the change it time
                    # from the start of the application, to its collection
                    csvRow += sample[col_index] - reference_time
                else:
                    csv_row += str(value)
                csv_row += delim
            csv_row = csv_row[:-1]
            csv_row += "\n"
            out_file.write(csv_row)

def output_rc_epochs(rc_data, directory=config.CSV_DIRECTORY):
    """Outputs a dictionary of epochs to csv files"""
    for row_col in rc_data:
        for epoch in row_col

    for rc_index in range(len(data_dict)):
        for epoch_index in range(len(data_dict[rc_index]["past"])):
            epoch = data_dict[rc_index]["past"][epoch_index]
            epoch.output_to_csv(epoch_index, directory=directory)

def output_epoch_average_list(avg_list, file_path, name_root, trial_num, p300_index):
    """Outputs the data held inside of the row/col_average lists to a .csv file"""
    for rc_index in range(len(avg_list)):
        rc = avg_list[rc_index]
        with open(file_path + name_root + "_" + str(rc_index) + "_" + str(trial_num) + ".csv", "w+") as outfile:
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
    """Selects element from list 'x' and returns it"""
    rand_index = random.randint(0, len(x) - 1)
    elem = x[rand_index]
    return elem

def get_desired_channels(sample, index_list):
    """Returns python list of desired channel data from an EEG sample"""
    new_sample = []
    for i in range(len(sample)):
        if i in index_list:
            new_sample.append(sample[i])
    return new_sample
    
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
    parser.add_argument('--gui-only',
                        dest="gui_only",
                        action='store_true',
                        default=False,
                        help='Only runs the gui in the live mode '
                        'without connecting to LSL data stream.')
    parser.add_argument('--output_raw',
                        dest="output_raw",
                        action='store_true',
                        help='Writes the raw data collected to a'
                        'CSV file to be used at a later time.')
    parser.add_argument('--output_epochs',
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
    #              Import and Set-up Classifier                #
    #==========================================================#

    classifier = None
    # Exit application if no classifier '.pkl' file can be found
    if args.live_mode:
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

    # 2D array of sample data [[read_time,c1,c2,...cn],...]
    data_history = np.zeros((0,1 + len(config.CHANNELS)), dtype=np.float64)
    
    # Sets up path for outputing the raw data
    if args.output_raw or args.output_epochs:
        output_dir = DIR_PATH + PATH_DELIM + str(config.CSV_DIRECTORY) + PATH_DELIM + datetime.datetime.now().strftime("%I-%M-%S%p%B_%d_%Y") + PATH_DELIM
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        data_filename = datetime.datetime.now().strftime("%I-%M-%S%p%B_%d_%Y") + "RawData.csv"

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

    # List of numpy matrices. Each matrix hold the average of the sample data for each channel
    
    row_averages = range(6)
    row_averages[0] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    row_averages[1] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    row_averages[2] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    row_averages[3] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    row_averages[4] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    row_averages[5] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages = range(6)
    col_averages[0] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages[1] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages[2] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages[3] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages[4] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)
    col_averages[5] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)), dtype=np.float64)

    # Holds all of the training examples
    X = np.zeros((0,SAMPLES_PER_EPOCH), dtype=np.float64)
    # Holds all of the example classifications
    y = np.array([])
    
    #==========================================================#
    #                    Spawn GUI Process                     #
    #==========================================================#
    
    # Create processes for the GUI and the EEG collection
    gui_process = Process(target=run_gui, args=(row_epoch_queue, col_epoch_queue, gui_conn, args.training_mode))
    # Start the GUI process
    gui_process.start()

    #==========================================================#
    #        Connect to LSL Stream or Simulation Data          #
    #==========================================================#

    app_start_time = pylsl.local_clock()

    if not args.gui_only and (args.live_mode or args.training_mode):
        # Look for the stream of EEG data on the network
        if args.verbose:
            print "Looking for EEG stream..."
        streams = resolve_stream('type', 'EEG')
        if args.verbose:
            print "Stream found!"
        # Create inlet to read from stream
        inlet = StreamInlet(streams[0])

    elif args.simulation_data_path != '':
        # Load the csv file at the given path
        infile = open(args.simulation_data_path, 'rb')
        data_reader = csv.reader(infile)
    
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
                    row_data[epoch.index]["active"].append(epoch)
                    row_epochs_read += 1
                    if first_epoch == None or epoch.start_time < first_epoch.start_time:
                        first_epoch = epoch
                    if last_epoch == None or epoch.start_time > last_epoch.start_time:
                        last_epoch = epoch

            if col_epochs_read < 6:
                epoch = col_epoch_queue.get_nowait()
                if epoch:
                    col_data[epoch.index]["active"].append(epoch)
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

        elif args.simulation_data_path != '':
            # Pull a row from the csv file and add it to the history
            try:
                sample_row = np.array(data_reader.next()).astype(np.float64)
                sample_row[0] = app_start_time + sample_row[0]
                data_history = np.vstack((data_history, sample_row))
            except:
                print "Ran out of simulation data"
                infile.close()
                # Break out of the data collection loop
                break
        

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

            
            if not args.gui_only:
                if args.simulation_data_path != '':
                    while time_stamp <= last_epoch.start_time + config.EPOCH_LENGTH + .5:
                            try:
                                sample_row = np.array(data_reader.next()).astype(np.float64)
                                sample_row[0] = app_start_time + sample_row[0]
                                data_history = np.vstack((data_history, sample_row))
                            except:
                                print "Ran out of simulation data"
                                infile.close()
                                # Break out of the data collection loop
                                break
                else:
                    # Get remaining data pertaining to last epoch
                    while time_stamp <= last_epoch.start_time + config.EPOCH_LENGTH + .5:
                        if args.verbose:
                            print "Getting remaining data from queue..."
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
                    write_raw_to_csv(data_history, app_start_time, output_dir + data_filename)
                    if args.verbose:
                            print "Getting remaining data from queue..."
                    
                # Erase any samples that fall outside of this sequence
                for row_index in range(len(data_history)):
                    t = data_history[row_index, 0] 
                    if t < first_epoch.start_time or t > last_epoch.start_time + config.EPOCH_LENGTH + .5:
                        data_history = np.delete(data_history, row_index, 0)
            
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
                            # Filter with lowpasss and high pass
                            data_history[:,column_index] = butter_lowpass_filter(data_history[:,column_index],
                                                                                config.FILTER_HIGHCUT,
                                                                                config.SAMPLING_RATE,
                                                                                order=config.FILTER_ORDER)
                            data_history[:,column_index] = butter_highpass_filter(data_history[:,column_index],
                                                                                config.FILTER_LOWCUT,
                                                                                config.SAMPLING_RATE,
                                                                                order=config.FILTER_ORDER)
                
            #==========================================================#
            #               Splitting Data into Epochs                 #
            #==========================================================#

            # Fill and move epochs
            if args.verbose:
                print "Splitting into epochs."
            for row in range(len(row_data)):
                for ep in row_data[row]["active"]:
                    ep.get_epoch_data(data_history)
                    row_data[row]["past"].append(ep)
                    row_data[row]["active"].remove(ep)
            for col in range(len(col_data)):
                for ep in col_data[col]["active"]:
                    ep.get_epoch_data(data_history)
                    col_data[col]["past"].append(ep)
                    col_data[col]["active"].remove(ep)
            if args.verbose:
                print "Done splitting."
            
            # Fill and move epochs
            if args.verbose:
                print "Splitting into epochs."
            for row in range(len(row_data)):
                for ep in row_data[row]["active"]:
                    ep.get_epoch_data(data_history)
                    row_data[row]["past"].append(ep)
                    row_data[row]["active"].remove(ep)
            for col in range(len(col_data)):
                for ep in col_data[col]["active"]:
                    ep.get_epoch_data(data_history)
                    col_data[col]["past"].append(ep)
                    col_data[col]["active"].remove(ep)
            if args.verbose:
                print "Done splitting."

            # Clear all data up to the start_time of last epoch
            data_history = np.zeros((0,1 + len(config.CHANNELS)), dtype=np.float64)

            #==========================================================#
            #                Averaging Together Epochs                 #
            #==========================================================#

            # Average the row data
            for row in row_data:
                for 

            # Average the rows
            for row in range(len(row_data)):
                # Sum the channel values
                for ep in row_data[row]["past"]:
                    data = ep.sample_data
                    for sample_num in range(len(data) - 1):
                        row_averages[row][sample_num,0:len(config.CHANNELS)] = row_averages[row][sample_num,0:len(config.CHANNELS)] + data[sample_num,0:len(config.CHANNELS)]
                # Divide by the # of epochs for the row
                row_averages[row] = row_averages[row] / len(row_data[row]["past"])

            # Average the columns
            for col in range(len(col_data)):
                # Sum the channel values
                for ep in col_data[col]["past"]:
                    data = ep.sample_data
                    for sample_num in range(len(data) - 1):
                        col_averages[col][sample_num,0:len(config.CHANNELS)] = col_averages[col][sample_num,0:len(config.CHANNELS)] + data[sample_num,0:len(config.CHANNELS)]
                # Divide by the # of epochs for the column
                col_averages[col] = col_averages[col] / len(col_data[col]["past"])

            
            if args.output_epochs and not args.gui_only:
                output_epoch_average_list(row_averages, output_dir, "Col", trials_complete, p300_col)
                output_epoch_average_list(col_averages, output_dir, "Row", trials_complete, p300_row)
            #==========================================================#
            #         Classification or Storage for Training           #
            #==========================================================#
            if args.live_mode:
                # Tuples for deicsion making (index of row/col, conf score,  # of positive p300 predictions)
                predicted_col = (-1, -1, -1)
                predicted_row = (-1, -1, -1)

                # pass row data from this past trial to the trained classifier
                for row_index in range(len(row_averages)):
                    prediction = classifier.predict(np.transpose(row_averages[row_index])).tolist()
                    confidence = classifier.decision_function(np.transpose(row_averages[row_index])).tolist()
                    print "Row # %d Prediction:" % row_index
                    print prediction
                    print "Confidence:"
                    print confidence
                    confidence_sum = np.sum(confidence)
                    false_count = prediction.count(0)
                    pos_count = prediction.count(1)
                    if  pos_count > predicted_row[2]:
                        predicted_row = (row_index, confidence_sum, pos_count)
                    elif confidence_sum > 0 and confidence_sum > predicted_row[1]:
                        predicted_row = (row_index, confidence_sum, pos_count)
                
                print ""
               
                # pass col data from this past trial to the trained classifier
                for col_index in range(len(col_averages)):
                    prediction = classifier.predict(np.transpose(col_averages[col_index])).tolist()
                    confidence = classifier.decision_function(np.transpose(col_averages[col_index]))
                    print "Col #%d Prediction:" % col_index
                    print prediction
                    print "Confidence:"
                    print confidence
                    confidence_sum = np.sum(confidence)
                    false_count = prediction.count(0)
                    pos_count = prediction.count(1)
                    if pos_count > predicted_col[2]:
                        predicted_col = (col_index, confidence_sum, pos_count)
                    elif confidence_sum > 0 and confidence_sum > predicted_col[1]:
                        predicted_col = (col_index, confidence_sum, pos_count)
                    
                main_conn.send(["prediction", "row", predicted_row[0]])
                main_conn.send(["prediction", "col", predicted_col[0]])

            else:
                # Add all row average data to the example lists
                for row_index in range(len(row_averages)):
                    for channel_index in range(len(row_averages[row_index][0])):
                        X = np.vstack((X, row_averages[row_index][:,channel_index]))
                        if row_index == p300_row:
                            y = np.append(y,1)
                        else:
                            y = np.append(y,0)
                # Add all col average data to the example lists
                for col_index in range(len(col_averages)):
                    for channel_index in range(len(col_averages[col_index][0])):
                        X = np.vstack((X, col_averages[col_index][:,channel_index]))
                        if col_index == p300_col:
                            y = np.append(y,1)
                        else:
                            y = np.append(y,0)             

            # Reset averagees to all zeros
            for i in range(6):
                row_averages[i] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)))
                col_averages[i] = np.zeros((SAMPLES_PER_EPOCH,len(config.CHANNELS)))
            
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
    elif args.simulation_data_path != '':
        infile.close()

    #==========================================================#
    #                         Other                            #
    #==========================================================#

    # Output the the epics to CSV files
    if args.output_epochs and not args.gui_only:

        """
        
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        if args.verbose:
            print "Writing output to: " + output_dir
        output_rc_epochs(row_data, directory=output_dir)
        output_rc_epochs(col_data, directory=output_dir)
        """
        """
        for col in range(len(col_data)):
            for i  in range(len(col_data[col]["past"]) - 1):
                ep = col_data[col]["past"][i]
                ep.output_to_csv(i, directory=output_dir)
        

        for row in range(len(row_data)):
            for i  in range(len(row_data[row]["past"]) - 1):
                ep = row_data[row]["past"][i]
                ep.output_to_csv(i, directory=output_dir)
        """
        if args.verbose:
            print "Done."
