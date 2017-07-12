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
from multiprocessing import Process, Manager, Queue
import datetime
import Tkinter as tk
# External library code
from pylsl import StreamInlet, resolve_stream
import numpy as np
# Custom modules
import butterwoth_bandpass
import speller_gui as gui
import config

class EEGEpoch(object):
    """Manages segments of EEG epoch data"""
    def __init__(self, is_row, index, start_time):
        self.is_row = is_row            # Boolean indicating if this epoch is for a row or column
        self.index = index              # Row/Col index of this epoch
        self.start_time = start_time    # Starting time of this epoch
        self.sample_data = []           # 2D array of samples of channel data

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
    
def run_gui(row_epoch_queue, col_epoch_queue):
    """Starts the p300 speller gui"""
    root = tk.Tk()
    root.title("P300 Speller")
    #root.geometry('%sx%s'%(config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    root.protocol("WM_DELETE_WINDOW", root.quit)
    speller_gui = gui.P300GUI(root,
                            row_epoch_queue,
                            col_epoch_queue)
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
                        default='live',
                        help='Runs the gui in training mode and '
                        'creates a new classifiers with the data')
    parser.add_argument('--gui_only',
                        metavar='gui_only',
                        type=str2bool,
                        default=False,
                        help='Only runs the gui in the live mode '
                        'without connecting to LSL data stream')
    args = parser.parse_args()

    #==========================================================#
    #          Set-up Queues for Managing Epochs               #
    #==========================================================#

    # Data that has been collected from the EEG
    manager = Manager()
    # Epochs that have been created by the GUI
    row_epoch_queue = Queue()
    col_epoch_queue = Queue()
    # Dictionary for row epochs
    row_data = {}
    row_data[0] = {"active":[], "past":[]}
    row_data[1] = {"active":[], "past":[]}
    row_data[2] = {"active":[], "past":[]}
    row_data[3] = {"active":[], "past":[]}
    row_data[4] = {"active":[], "past":[]}
    row_data[5] = {"active":[], "past":[]}
    # Dictionary for column epochs
    col_data = {}
    col_data[0] = {"active":[], "past":[]}
    col_data[1] = {"active":[], "past":[]}
    col_data[2] = {"active":[], "past":[]}
    col_data[3] = {"active":[], "past":[]}
    col_data[4] = {"active":[], "past":[]}
    col_data[5] = {"active":[], "past":[]}

    # Dictionary of data samples collected in this sequence (time_stamp -> channel_data)
    data_history = {}
    
    # Create processes for the GUI and the EEG collection
    gui_process = Process(target=run_gui, args=(row_epoch_queue, col_epoch_queue))
    # Start the GUI process
    gui_process.start()

    if not args.gui_only:
        # Look for the stream of EEG data on the network
        print "Looking for EEG stream..."
        streams = resolve_stream('type', 'EEG')
        print "Stream found!"
        # Create inlet to read from stream
        inlet = StreamInlet(streams[0])
    
    #==========================================================#
    #             DATA COLLECTION AND MANIPULATION             #
    #==========================================================#

    trials_complete = 0
    sequences_complete = 0
    row_epochs_read = 0
    col_epochs_read = 0

    # Run data collection for as long as the gui is active
    while gui_process.is_alive():
        # reference to the most recent epoch
        epoch = None

        if not args.gui_only:
            # Get sample and time_stamp from data stream
            sample, time_stamp = inlet.pull_sample(timeout=0.0)
            # Store the sample locally
            if sample != None:
                update_data_history(data_history, time_stamp, sample)
        
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
            # Catch any exception that is thrown as a result of trying
            # to read from an empty queue
            pass
        
        # Sequence complete
        if col_epochs_read == 6 and row_epochs_read == 6:
            sequences_complete += 1
            col_epochs_read = 0
            row_epochs_read = 0


        # Trial complete
        if sequences_complete >= config.SEQ_PER_TRIAL:

            # Get remaining data pertaining to last epoch
            if not args.gui_only:
                while time_stamp != None and time_stamp <= epoch.start_time + config.EPOCH_LENGTH:
                    print "Getting remaining data from queue..."
                    sample, time_stamp = inlet.pull_sample(timeout=0.1)
                    update_data_history(time_stamp, sample)

            # Run bandpass filter on the data
            # Move all data to an anp arry
    
            print "Splitting into epochs."
            # Fill and move epochs
            for row in row_data.keys():
                for ep in row_data[row]["active"]:
                    ep.sample_data = get_epoch_data(ep.start_time, data_history)
                    row_data[row]["past"].append(ep)
                    row_data[row]["active"].remove(ep)
            for col in col_data.keys():
                for ep in col_data[col]["active"]:
                    ep.sample_data = get_epoch_data(ep.start_time, data_history)
                    col_data[col]["past"].append(ep)
                    col_data[col]["active"].remove(ep)
            print "Done splitting."

            # Clear all data up to the start_time of last epoch
            for key in sorted(data_history.keys()):
                if key <= epoch.start_time:
                    del data_history[key]

            # Average data from the epochs
            row_averages = {}
            row_averages[0] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            row_averages[1] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            row_averages[2] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            row_averages[3] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            row_averages[4] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            row_averages[5] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages = {}
            col_averages[0] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages[1] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages[2] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages[3] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages[4] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))
            col_averages[5] = np.zeros((samples_per_epoch,config.NUM_CHANNELS))

            # Average the rows
            for row in row_data.keys():
                # Sum the channel values
                for ep in row_data[row]["past"]:
                    data = np.array(ep.sample_data);
                    for sample_num in range(len(data)):
                        row_averages[row][sample_num,0:config.NUM_CHANNELS - 1] =
                            row_averages[row][sample_num,0:config.NUM_CHANNELS - 1] +
                            data[sample_num,0:config.NUM_CHANNELS - 1]
                # Divide by the # of epochs for the row
                row_averages[row] = row_averages[row] / len(row_data[row]["past"])

            # Average the columns
            for col in c0l_data.keys():
                # Sum the channel values
                for ep in col_data[col]["past"]:
                    data = np.array(ep.sample_data);
                    for sample_num in range(len(data)):
                        col_averages[col][sample_num,0:config.NUM_CHANNELS - 1] =
                            col_averages[col][sample_num,0:config.NUM_CHANNELS - 1] +
                            data[sample_num,0:config.NUM_CHANNELS - 1]
                # Divide by the # of epochs for the column
                col_averages[col] = col_averages[col] / len(col_data[col]["past"])


            if args.mode = 'train':

            else:
                pass
            
            
            
           

            
            # Clear out all the data samples that occurred before the start
            # of the last epoch
            
            
            
            # Reset counter
            sequences_complete = 0

    # Output the the epics to CSV files
    if config.OUTPUT_CSV:
        

        dir_path = os.path.dirname(os.path.realpath(__file__))
        output_dir = dir_path + "\\" + str(config.CSV_DIRECTORY) + "\\" + datetime.datetime.now().strftime("%I-%M-%S%p%B_%d_%Y") + "\\"

        os.mkdir(output_dir)
        print "Writing output to: " + output_dir

        for col in col_data.keys():
            for i  in range(len(col_data[col]["past"]) - 1):
                ep = col_data[col]["past"][i]
                ep.output_to_csv(i, directory=output_dir)

        for row in row_data.keys():
            for i  in range(len(row_data[row]["past"]) - 1):
                ep = row_data[row]["past"][i]
                ep.output_to_csv(i, directory=output_dir)
        print "Done."