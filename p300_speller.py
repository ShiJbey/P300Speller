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
from multiprocessing import Process, Manager, Value
import numpy as np
import copy

# Default port to be used
PORT = '/dev/ttyUSB0'
# OpenBCI Board Reference
BCI_BOARD = None

# Create a manager for keeping track of sample data and gui updates
manager = Manager()
    
# Dictionary of times mapped to samples
DATA_HISTORY = manager.dict()
update_times = manager.list()

# Thread-safe dictionary for row epochs
ROW_DATA = manager.dict()
# Thread-safe dictionary for col epochs
COL_DATA = manager.dict()

EPOCH_LENGTH = 1 #time in seconds

class EEGEpoch:
    """Manages EEG epoch data"""
    
    def __init__(self, start_time):
        self.start_time = start_time
        # 2D array of samples and channel data
        self.sample_data = []
        # Are we collecting data for this Epoch
        self.active = True

    def is_within_epoch(self, time):
        """Returns true if the given time is within this given epoch"""
        return (time >= self.start_time) and (time <= self.start_time + EPOCH_LENGTH)
        
    def update(self, sample_data):
        """Appends the given sample data to the list of channel_data"""
        self.sample_data.append(sample_data)

    def average(self):
        arr = np.array(self.sample_data)
        self.sample_data = np.mean(arr, axis=0)

    def __str__(self):
        return "Start: %s \nActive?: %s \nNum Samples: %s" %(self.start_time,self.active, len(self.sample_data))
 

def get_epoch(start_time):
    """
    Loops through the data history and returns an epoch Object with the
    data that fell within this epoch
    """
    #print("Length of data in get_epoch: %d" %(len(DATA_HISTORY)))
    epoch = EEGEpoch(start_time)
    for k in DATA_HISTORY.keys():
        if (epoch.is_within_epoch(k)):
            epoch.update(copy.copy(DATA_HISTORY[k]))
    #print("Length of data in get_epoph epoch: %d" %(len(epoch.sample_data)))
    return epoch

def print_data(sample):
    """Prints the data within the sample"""
    if (BCI_BOARD.streaming == True):
        print(vars(sample))

def update_data_history(sample):
    """Adds the sample channel data to the dict under its read time"""
    # Adds all data samples to the dict under read times
    DATA_HISTORY[sample.read_time] = sample.channel_data
    """
    for index in ROW_DATA.keys():
        row = ROW_DATA[index]
        for epoch in row["active"]:
            if (epoch.is_within_epoch(sample.read_time)):
                #epoch.update(sample.channel_data)
            else:
                epoch.active = False
                #epoch.sample_data = get_epoch(epoch.start_time).sample_data
                #epoch.average()
                row["past"].append(epoch)
                row["active"].remove(epoch)
        ROW_DATA[index] = row

    for index in COL_DATA.keys():
        col = ROW_DATA[index]
        for epoch in col["active"]:
            if (epoch.is_within_epoch(sample.read_time)):
                #epoch.update(sample.channel_data)
            else:
                epoch.active = False
                #epoch.sample_data = get_epoch(epoch.start_time).sample_data
                #epoch.average()
                col["past"].append(epoch)
                col["active"].remove(epoch)
        COL_DATA[index] = col
    """
    
def run_eeg(bci_board, history_dict):
    """Starts streaming data from the OpenBCI board and updating the history"""
    logging.basicConfig(filename="p300SpellerLog.txt",
        format='%(asctime)s - %(levelname)s : %(message)s',
        level=logging.DEBUG)
    logging.info('============== LOG START ==============')
    print("Board Connected")
    bci_board.ser.write('v')
    time.sleep(5)
    bci_board.start_streaming(update_data_history)
    
def run_gui(row_data, col_data, rect_update_rate_millis=100):
    """Starts the p300 speller gui"""
    root = tk.Tk()
    root.title("P300 Speller")
    root.protocol("WM_DELETE_WINDOW", root.quit)
    # Pass the epoch data dicts to the gui for updating
    speller_gui = gui.P300GUI(root, row_data, col_data, update_rate=rect_update_rate_millis)
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
        PORT =  str(sys.argv[1])

    # Create reference to the OpenBCI board
    BCI_BOARD = bci.OpenBCIBoard(port=PORT, scaled_output=False, log=True)


    ROW_DATA[0] = manager.dict({"active":[], "past":[]});
    ROW_DATA[1] = manager.dict({"active":[], "past":[]});
    ROW_DATA[2] = manager.dict({"active":[], "past":[]});
    ROW_DATA[3] = manager.dict({"active":[], "past":[]});
    ROW_DATA[4] = manager.dict({"active":[], "past":[]});
    ROW_DATA[5] = manager.dict({"active":[], "past":[]});

    # Thread-safe dictionary for column epochs
    
    COL_DATA[0] = {"active":[], "past":[]}; COL_DATA[1] = {"active":[], "past":[]};
    COL_DATA[2] = {"active":[], "past":[]}; COL_DATA[3] = {"active":[], "past":[]};
    COL_DATA[4] = {"active":[], "past":[]}; COL_DATA[5] = {"active":[], "past":[]};

    # Create processes for the GUI and the EEG collection
    eeg_process = Process(target=run_eeg, args=(BCI_BOARD, DATA_HISTORY))
    eeg_process.daemon = True
    gui_process = Process(target=run_gui, args=(ROW_DATA,COL_DATA))

    # Start processes
    eeg_process.start()
    gui_process.start()

    # Wait for gui process to finish
    gui_process.join()
    
    # Now we set the board to stop streaming
    print("Stopping board")
    BCI_BOARD.stop()
    BCI_BOARD.disconnect()

    # ====== TESTING EPOCH SEPARATION ======
    print(len(DATA_HISTORY))

    # Fill all the epochs
    for row in ROW_DATA.keys():
        for status in ROW_DATA[row].keys():
           for i, ep in enumerate(ROW_DATA[row][status]):
               all_data = get_epoch(ep.start_time).sample_data
               ep.sample_data = all_data[:]
               print(ep)
    
    """
    # Fill epochs with data samples
    try:
        it = iter(ROW_DATA[0]["active"])
        while(1):
            ep = it.next()
            all_data = copy.copy(get_epoch(ep.start_time).sample_data)
            ep.sample_data = all_data
            #print(ep)
    except StopIteration:
        pass
    """
    print("\n\n")

    # Print epochs for the first row
    for row in ROW_DATA.keys():
        for status in ROW_DATA[row].keys():
           for i, ep in enumerate(ROW_DATA[row][status]):
               print (ep)
    
