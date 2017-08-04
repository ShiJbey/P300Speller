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
from build_classifier import *
import eeg_event_codes
import config

def get_closest_event_to_time(target_time, events, tolerance=None):
    """
    Returns the event tuple that occured closest to the target time
    """
    closest_event = None
    if not tolerance:
        for event in events:
            if event[0] <= target_time:
                closest_event = event
            else:
                break
    else:
        for event in events:
            if (event[0] >= target_time - tolerance and
                event[0] <= target_time + tolerance):
                closest_event = event
            elif event[0] > target_time + tolerance:
                break
    return closest_event

def add_events_to_data_history(data_history, events):
    """
    Fills in the second column of the data history matrix
    with the corect event codes and returns the new matrix
    """
    row_length =  int(np.shape(data_history)[1])
    filled_data_history = np.zeros((0, row_length))

    events = sorted(events)
    for row in data_history:
        closest_event = get_closest_event_to_time(row[0], events)
        if closest_event:
            row[1] = closest_event[1]
            events.remove(closest_event)
        filled_data_history = np.vstack((filled_data_history, row)) 
    return filled_data_history

def write_raw_to_csv(data_history, file_path, delim=","):
    """
    Given an array of sample data arrays, writes the data to a file
    """
    # Sort the epoch times
    with open(str(file_path), 'a+') as out_file:
        for sample_row in data_history:
            csv_row = ""
            for value in sample_row:
                csv_row += str(repr(value))
                csv_row += delim
            csv_row = csv_row[:-1]
            csv_row += "\n"
            out_file.write(csv_row)

def trim_data_history(data_history, start_time, end_time, num_channels=len(config.EEG_CHANNELS)):
    """
    Returns a matrix of data exluding any samples that fell
    outside of the time interval
    """
    trimmed_history = np.zeros((0,2 + num_channels), dtype=np.float64)
    for sample_row in data_history:
        sample_time = sample_row[0]
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

def get_p300_prediction(clf, average_data, sampling_rate=config.SAMPLING_RATE):
    """
    Makes a prediction on a row/column index given a classifier and list of sample data matrices
    """
    best_prediction = {"index": 0, "confidence": 0.0}

    for index in range(len(average_data)):
        data = np.ravel(average_data[index])
        predicted_class = clf.predict(np.reshape(data,(1, -1)))
        prediction_confidence = clf.predict_proba(np.reshape(data,(1, -1)))[0][1]
        #prediction_confidence = clf.decision_function(np.reshape(data,(1, -1)))[0]
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
    import mne

    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    PATH_DELIM = '/'
    if os.name == 'nt':
        PATH_DELIM = '\\'
    
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
                        default=False,
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

    if args.gui_only:
        args.training_mode = False
        args.live_mode = False

    if args.live_mode:
        args.training_mode = False

    if args.live_mode and args.training_mode:
        raise parser.error("Only one mode (train or live) may be specified at a time.")

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
    RAW_DATA_FILENAME = "RawData.csv"
    
    if not os.path.exists(OUTPUT_DIR):
        if args.verbose:
            print "Output directory does not exist"
            print "Creating output directory at: %s" % (OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    #==========================================================#
    #                  Set-up Data Structures                  #
    #==========================================================#

    # Queues for epochs that have been created by the GUI
    event_queue = Queue()

    # Pipe to send a predicted character to the gui to update the spelling buffer
    main_conn, gui_conn = Pipe()

    # 2D numpy array of sample data [[read_time,c1,c2,...cn],...]
    data_history = np.zeros((0,2 + len(config.EEG_CHANNELS)), dtype=np.float64)

    # List of times of all the epochs in the current trial
    events = []

    #==========================================================#
    #                         Constants                        #
    #==========================================================#

    EEG_LABELS = ['time-stamps', 'event'] + config.EEG_CHANNELS
    EEG_TYPE_LABELS = ['misc', 'stim'] + config.EEG_TYPES

    EEG_INFO = mne.create_info(EEG_LABELS,
                               config.SAMPLING_RATE,
                               ch_types=EEG_TYPE_LABELS)
    
    #==========================================================#
    #                    Spawn GUI Process                     #
    #==========================================================#
    
    # Create processes for the GUI and the EEG collection
    gui_process = Process(target=gui.run_gui, args=(event_queue, gui_conn, args.training_mode, args.verbose))
    # Start the GUI process
    gui_process.start()

    #==========================================================#
    #        Connect to LSL Stream or Simulation Data          #
    #==========================================================#

    if args.live_mode or args.training_mode:
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
    events_read = 0
    p300_row = 0              # Row that has been randomly selected for training
    p300_col = 0              # Column that has been randomly selected for training
    epochs = None
    first_epoch_time = -1
    last_epoch_time = -1

    if args.training_mode:
        # Start traing at the letter A
        main_conn.send(["train", (p300_row, p300_col)])

    # Run data collection for as long as the gui is active
    while gui_process.is_alive():
        
        #==========================================================#
        #            Read Epochs Created by the GUI                #
        #==========================================================#

        try:
            if events_read < 12:
                event = event_queue.get_nowait()
                if event:
                    events.append(event)
                    events_read += 1
                    if events_read == 1:
                        first_epoch_time = event[0]
                    elif events_read == 12:
                        last_epoch_time = event[0]
        except:
            pass

        #==========================================================#
        #                 Read Sample From LSL                     #
        #==========================================================#

        if args.live_mode or args.training_mode:
            # Get sample and time_stamp from data stream
            sample, time_stamp = inlet.pull_sample(timeout=0.01)
            
            if sample != None:
                # Store the sample in the data history
                sample_row = np.append(time_stamp, np.append(eeg_event_codes.NO_EVENT, sample[0:len(config.EEG_CHANNELS)]))
                data_history = np.vstack((data_history, sample_row))
        

        #==========================================================#
        #          Processing the End of the Sequence              #
        #==========================================================#
        
        # Sequence complete
        if events_read == 12:
            sequences_complete += 1
            events_read = 0
            if np.shape(data_history)[0] == 0 and not args.gui_only:
                print "ERROR: COMPLETED TRIAL WITHOUT ANY DATA"

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
                    if args.verbose:
                        print "Getting remaining data from queue..."
                        
                    # Get remaining data pertaining to last epoch
                    while time_stamp <= last_epoch_time + config.EPOCH_LENGTH + .5:
                        sample, time_stamp = inlet.pull_sample(timeout=0.1)
                        if sample != None:
                            sample_row = np.append(time_stamp, np.append(eeg_event_codes.NO_EVENT, sample[0:len(config.EEG_CHANNELS)]))                           
                            data_history = np.vstack((data_history, sample_row))
                        else:
                            break
                    
                    if args.verbose:
                        print "Adding event codes to the data history"
                    # Add the events to the data history
                    data_history = add_events_to_data_history(data_history, events)                  
                    events = []

                    if args.output_raw or args.training_mode:
                        # Write the contents of the
                        write_raw_to_csv(data_history, OUTPUT_DIR + RAW_DATA_FILENAME)
                        if args.verbose:
                                print "Wrote raw data to \'%s\' in the output directory." % (RAW_DATA_FILENAME)

            #==========================================================#
            #        MNE Raw Data filtering and Epoch Processing       #
            #==========================================================#

            if args.live_mode or args.training_mode:
                # Create raw data array in MNE
                mne_raw_data = mne.io.RawArray(np.transpose(data_history), EEG_INFO)            

                if args.verbose:
                    print "Filtering the raw data"
                
                print np.shape(data_history)
                filtered_data = mne_raw_data.filter(0.1, h_freq=None, l_trans_bandwidth = 0.05)
                del data_history
                data_history = np.zeros((0,2 + len(config.EEG_CHANNELS)), dtype=np.float64)
                filtered_data = filtered_data.filter( l_freq=None, h_freq=40)
            
                # Get MNE events
                if args.verbose:
                    print "Extracting events from data"
                mne_events = mne.find_events(filtered_data, stim_channel='event', shortest_event=1, consecutive=True)
                filtered_data.drop_channels(['event'])

                if args.verbose:
                    print "Splitting data into epochs"
                
                # Make a copy of the event ID and filter out event code that were not seen
                if args.training_mode:
                    event_ids = eeg_event_codes.get_event_id_dict(p300_row, p300_col)
                else:
                    event_ids = eeg_event_codes.LIVE_EVENT_ID

                epochs = mne.Epochs(filtered_data, mne_events, event_id=event_ids, add_eeg_ref=False, tmin = 0, tmax=1, decim=config.DOWN_SAMPLE_FACTOR)
                epochs.load_data()

                

            #==========================================================#
            #                Averaging Together Epochs                 #
            #==========================================================#

            if args.live_mode or args.training_mode:

                if args.verbose:
                    print "Averaging the epochs for each row/column"

                all_row_averages = [
                    np.transpose(epochs['row0'].average().data),
                    np.transpose(epochs['row1'].average().data),
                    np.transpose(epochs['row2'].average().data),
                    np.transpose(epochs['row3'].average().data),
                    np.transpose(epochs['row4'].average().data),
                    np.transpose(epochs['row5'].average().data)
                ]

                all_col_averages = [
                    np.transpose(epochs['col0'].average().data),
                    np.transpose(epochs['col1'].average().data),
                    np.transpose(epochs['col2'].average().data),
                    np.transpose(epochs['col3'].average().data),
                    np.transpose(epochs['col4'].average().data),
                    np.transpose(epochs['col5'].average().data)
                ]

                svc_row_averages = [
                    np.transpose(epochs['row0'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['row1'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['row2'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['row3'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['row4'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['row5'].pick_channels(config.CHANNELS_FOR_SVC).average().data)
                ]

                svc_col_averages = [
                    np.transpose(epochs['col0'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['col1'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['col2'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['col3'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['col4'].pick_channels(config.CHANNELS_FOR_SVC).average().data),
                    np.transpose(epochs['col5'].pick_channels(config.CHANNELS_FOR_SVC).average().data)
                ]

               
                if args.training_mode:
                    if args.verbose:
                        print "Outputing epoch averages for this trial"
                    output_averaged_epoch_list(all_row_averages, OUTPUT_DIR, "Col", trials_complete, p300_col)
                    output_averaged_epoch_list(all_col_averages, OUTPUT_DIR, "Row", trials_complete, p300_row)
            
            #==========================================================#
            #                     Classification                       #
            #==========================================================#
            if args.live_mode:

                if args.verbose:
                    print "Making prediction..."

                predicted_row = get_p300_prediction(classifier, svc_row_averages)
                predicted_col = get_p300_prediction(classifier, svc_col_averages)

                if args.verbose:
                    print "Classifier predicted Row: %d and Col: %d" % (predicted_row, predicted_col)
                    
                main_conn.send(["prediction", (predicted_row, predicted_col)])

            #==========================================================#
            #                 Prep for the Next Trial                  #
            #==========================================================#

            # Reset counter
            sequences_complete = 0
            del epochs
            first_epoch_time = -1
            last_epoch_time = -1
            del mne_events
            del filtered_data
            del mne_raw_data
            events = []

            if args.training_mode:
                # Increment the column
                p300_col = (p300_col + 1) % 6
                if p300_col == 0:
                    # Increment the row when we get to the end of a row
                    p300_row = (p300_row + 1) % 6
                # Send these over the pipe to the GUI
                main_conn.send(["train", (p300_row, p300_col)])
                
    #==========================================================#
    #                 Disconnect LSL Stream                    #
    #==========================================================#

    if args.live_mode or args.training_mode:
        inlet.close_stream()
        if args.verbose:
            print "Closed data stream."