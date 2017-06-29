#! /usr/bin/python

import sys
sys.path.append('./OpenBCI_Python')
import speller_gui as gui
import open_bci_v3 as bci
import serial
import os
import logging
import time
import threading
import Tkinter as tk
from multiprocessing import Process, Manager, Value, Queue
import numpy as np
import copy
import config
import csv


data_history = None

class EEGEpoch:
    """Manages segments of EEG epoch data"""
    
    def __init__(self, is_row, index, start_time):
        self.is_row = is_row
        self.index = index
        self.start_time = start_time
        # 2D array of samples and channel data
        self.sample_data = []
        # Are we collecting data for this Epoch
        self.active = True

    def is_within_epoch(self, time):
        """Returns true if the given time is within this given epoch"""
        return (time >= self.start_time) and (time <= self.start_time + config.EPOCH_LENGTH)

    def __str__(self):
        return "Start: %s \nActive?: %s \nNum Samples: %s" %(self.start_time, self.active, len(self.sample_data))

    def output_to_csv(self, epoch_num, delim=","):
        """Output the data held by this epoch to a csv file"""
        filename = ""
        if self.is_row:
            filename += "Row_"
        else:
            filename += "Col_"
        filename += str(self.index) + "_epoch" + "_" + str(epoch_num) + ".csv"

        # First line has the starting time
        with open("./" + str(config.CSV_DIRECTORY) + str(filename), 'a') as f:
            f.write(str(repr(self.start_time)) + "\n")
            # All following lines are organized one sample per
            #  row and each column is a different channel
            for sample in self.sample_data:
                row = ""
                for voltage in sample:
                    row += str(voltage)
                    row += delim
                row = row[:-1]
                row += "\n"
                f.write(row)
            

def get_epoch_data(start_time, data_history):
    """Loops through the data history and returns a 2D array of sample data"""
    # 2D array that will hold channel data from samples
    data = []

    # Loop through the list of keys until we find the sample
    #  closest to the start time
    closest_time = -1
    for read_time in sorted(data_history.keys()):
        if (read_time <= start_time):
            closest_time = read_time
        elif (read_time > start_time):
            if (closest_time == -1):
                closest_time = read_time
            break

    # Get all 250 samples for the epic
    start_sample_index = sorted(data_history.keys()).index(closest_time)
    samples_per_epoch = config.EPOCH_LENGTH * 250
    sample_times = sorted(data_history.keys())[start_sample_index:start_sample_index + (samples_per_epoch + 1)] 

    # Append the sample data for all the samples
    for time in sample_times:
        data.append(data_history[time])
        
    return data

def update_data_history(sample):
    """Adds the sample channel data to the dict under its read time"""
    data_history[sample.read_time] = sample.channel_data
    for i in range(4):
        if (sample.channel_data[i] <= -8000000):
            print("Channel %d has lost connection..." %(i+1))

def run_eeg(bci_board):
    """Starts streaming data from the OpenBCI board and updating the history"""
    logging.basicConfig(filename=config.LOG_FILENAME,
        format='%(asctime)s - %(levelname)s : %(message)s',
        level=logging.DEBUG)
    logging.info('============== LOG START ==============')
    print("Board Connected")
    bci_board.ser.write(b'v')
    #time.sleep(0.100)
    #bci_board.ser.write(b'[')
    time.sleep(5)
    bci_board.start_streaming(update_data_history)
    
def run_gui(row_epoch_data, col_epoch_data):
    """Starts the p300 speller gui"""
    root = tk.Tk()
    root.title("P300 Speller")
    root.protocol("WM_DELETE_WINDOW", root.quit)
    speller_gui = gui.P300GUI(root,
            row_epoch_queue,
            col_epoch_queue,
            highlight_time=config.HIGHLIGHT_TIME,
            intermediate_time=config.INTERMEDIATE_TIME)
    root.mainloop()

"""
This module runs the p-300 speller using multiple processes.
Processes are spawned for the main module, the eeg data
collection, and the GUI. The eeg_process is responsible for
sending sample data to the data buffer managed by the main module.
The gui process is responsible for updating the gui and recording
times when the gui has been updated
"""
if __name__ == '__main__':
    # Change default port if a new one was given
    if (len(sys.argv) == 2):
        config.PORT =  str(sys.argv[1])


    # Data that has been collected from the EEG
    manager = Manager()
    data_history = manager.dict()

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

    # Create reference to the OpenBCI board
    bci_board = bci.OpenBCIBoard(port=config.PORT, scaled_output=False, log=config.LOG_OUTPUT)

    # Create processes for the GUI and the EEG collection
    eeg_process = Process(target=run_eeg, args=(bci_board,))
    eeg_process.daemon = True
    gui_process = Process(target=run_gui, args=(row_epoch_queue, col_epoch_queue))

    # Start processes
    eeg_process.start()
    gui_process.start()

    num_iters =  0;
    # Read from the queues
    while(gui_process.is_alive()):
        try:
            epoch = row_epoch_queue.get_nowait()
            if (epoch):
                row_data[epoch.index]["active"].append(epoch)
        
            epoch = col_epoch_queue.get_nowait()
            if (epoch):
                col_data[epoch.index]["active"].append(epoch)
                if (epoch.index == 5):
                    num_iters += 1
        except:
            pass

        if (num_iters >= config.ITERS_BETWEEN_ANALYSIS):
            # Sleep to allow the last epoch's data to be
            # collected
            time.sleep(config.EPOCH_LENGTH)
            # Get data for all epochs
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

            print("Moved Epochs")
            # clear half of data history
            for key in sorted(data_history.keys())[:500]:
                del data_history[key]
            
            # Reset counter
            num_iters = 0

    
    # Set the board to stop streaming
    print("Stopping board")
    bci_board.stop()
    bci_board.disconnect()

    # Output the the epics to CSV files
    if (config.OUTPUT_CSV):
        for col in col_data.keys():
            for i  in range(len(col_data[col]["past"]) - 1):
                if (i > 0):
                    ep = col_data[col]["past"][i]
                    ep.output_to_csv(i)

        for row in row_data.keys():
            for i  in range(len(row_data[row]["past"]) - 1):
                if (i > 0):
                    ep = row_data[row]["past"][i]
                    ep.output_to_csv(i)


