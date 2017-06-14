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

   
class P300Speller:
    """
    P300Speller allows some one to spell via the
    use of an OpenBCI.
    This p300 speller application runs as 3 threads:
    a main, an EEGProcessingThread, and a GUIThread.
    """
    
    def __init__(self, eeg_thread=None, gui_thread=None):
        self.spelling_buffer = ""       # Buffer where selected characters are concatenated
        self.eeg_thread = eeg_thread    # Reference to the EEGProcessingThread
        self.gui_thread = gui_thread    # Reference to the GUIThread
        self.select_rect_history = []   # History of past selection rect info ("orientation",pos,time)
        self.rect_history_lock = threading.Lock()   # Synchronization lock
        self.eeg_data_history = []                  # Not sure how this is going to be stored
        self.eeg_data_lock = threading.Lock()       # Synchronization lock
        self.estimated_position = (-1,-1)           # Estimated position of the desired character
        self.row_trials = 0     # Number of row trials for this estimation
        self.col_trials = 0     # Number of column trials for this esitmation
    
    def set_gui_thread(self, thread):
        self.gui_thread = thread
        thread.speller = self

    def set_eeg_thread(self, thread):
        self.eeg_thread = thread
        thread.speller = self

    def start(self):
        """Starts all threads"""
        if (self.eeg_thread):
            self.eeg_thread.start()
        if (self.gui_thread):
            self.gui_thread.start()

        # Program runs while gui is active
        while self.gui_thread.is_alive():
            x = 1 + 1
        
        # Close the program
        self.stop()

    def stop(self):
        print("Main is closing app")
        self.eeg_thread.stop()
        self.eeg_thread.join()
        exit()

class EEGProcessingThread(threading.Thread):
    """Thread responsible for running the EEG processing """

    def __init__(self, thread_id, name, bci_board, speller=None):
        super(EEGProcessingThread, self).__init__()
        self.thread_id = thread_id
        self.name = name
        self.speller = speller # Reference to the P300Speller in the main
        self.bci_board = bci_board
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.isSet()
    
    def print_data(self, sample):
        if self.stopped():
            seld.bci_board.streaming = False
            #self.bci_board.ser.write('s')
        print(sample.channel_data)
 

    def run(self):
        baud = 115200
        logging.basicConfig(filename="p300SpellerLog.txt",
            format='%(asctime)s - %(levelname)s : %(message)s',
            level=logging.DEBUG)
        logging.info('============== LOG START ==============')
        print("Board Connected")
        self.bci_board.ser.write('v')
        time.sleep(10)
        self.bci_board.start_streaming(self.print_data)

# Running the GUI in its own thread free's up the main application
class GUIThread(threading.Thread):
    """Thread that runs the GUI"""

    def __init__(self, thread_id, name, speller=None,
            rect_update_rate_milis=10, suggestion1_flash_rate=3,
            suggestion2_flash_rate=1, suggestion3_flash_rate=5):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = self.name
        self.speller = speller
        self.speller_gui = None;
        self.root = None;

    def exitApp(self):
        print("Closing App")
        self.root.quit()
        self.speller.stop()

    def run(self):
        print("Running GUI")
        self.root = tk.Tk()
        self.root.title("P300 Speller")
        self.root.protocol("WM_DELETE_WINDOW", self.exitApp)
        self.speller_gui = gui.P300GUI(self.root, self)
        self.root.mainloop()



# We should either:
# 1) Wait for the connection for the board to start the GUI
# 2) Start the GUI, but do not issue updates until connection found
# 3) Start the GUI, and have an intermediate screen that has a
#    button that becomes active only when the connection has been
#    established.
#
# Option 1, is implemented below

if __name__ == '__main__':
    port = '/dev/ttyUSB0'
    
    speller = P300Speller()
        
    speller.set_eeg_thread(EEGProcessingThread(1, "open-bci", bci.OpenBCIBoard(port=port, scaled_output=False, log=True)))
    
    
    speller.set_gui_thread(GUIThread(2,"gui"))
    #speller.gui_thread.start()
    #speller.eeg_thread.start()
    speller.start()
    
    """
    except:
        # Catches the exeption thrown when the bci board
        # can't be found
        print("Could not connect to OpenBCI board")
    """




