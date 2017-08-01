"""
Epoch objects are resposnsible for segmenting time-chunks
of EEG data that have been collected
"""
import numpy as np
from eeg_event_codes import encode_event, decode_event


def get_epochs_from_codes(code_time_pairs):
    """
    Given a list of event codes, returns a list of epochs
    """
    epochs = []
    for event in code_time_pairs:
        if event[1] != 32:
            epochs.append(Epoch.create_epoch_from_code(int(event[1]), event[0]))
    return epochs

def separate_epochs_by_orientation(epoch_list):
    """
    Given a list of epochs returns two epoch
    lists. One with all row epochs and the other
    with all column epochs
    """
    row_epochs = []
    col_epochs = []
    for epoch in epoch_list:
        if epoch.is_row:
            row_epochs.append(epoch)
        else:
            col_epochs.append(epoch)
    return row_epochs, col_epochs

def separate_epochs_by_index(epoch_list):
    """
    Given a list of epochs, returns a matrix of epochs
    separated by their indices
    """
    sorted_epochs = [[], [], [], [], [], []]

    for epoch in epoch_list:
        sorted_epochs[epoch.index].append(epoch)

    return sorted_epochs

def generate_epoch_matrices(code_time_pairs):
    """
    Given a list of event codes, returns two matrices of epochs
    """
    epochs = get_epochs_from_codes(code_time_pairs)

    row_epoch_list, col_epoch_list = separate_epochs_by_orientation(epochs)

    row_epoch_matrix = separate_epochs_by_index(row_epoch_list)
    col_epoch_matrix = separate_epochs_by_index(col_epoch_list)

    return row_epoch_matrix, col_epoch_matrix

def get_epoch_matrix_data(epoch_matrix, data_history, samples_per_epoch):
    """
    Gets the epoch data for each epoch in the matrix
    """
    print "Getting data"
    for matrix_row in epoch_matrix:
        for epoch in matrix_row:
            epoch.get_epoch_data(data_history, samples_per_epoch)

def average_epochs(epoch_list, samples_per_epoch, num_channels):
    """
    Given a list of Epochs, returns 2D numpy
    of the averaged sample data
    """
    average_data = np.zeros((samples_per_epoch, num_channels), dtype=np.float64)

    for epoch in epoch_list:
        data = epoch.sample_data
        for sample_num in range(len(data)):
            average_data[sample_num, :] = average_data[sample_num, :] + data[sample_num, :]

    average_data = average_data / len(epoch_list)

    return average_data

def average_epoch_matrix(epoch_matrix, samples_per_epoch, num_channels):
    """
    Given a list of lists (matrix) of Epochs
    returns a list of numpy 2D arrays of the average
    epoch data for each row/col
    """
    averaged_epoch_data = []

    for epoch_list in epoch_matrix:
        averaged_epoch_data.append(average_epochs(epoch_list, samples_per_epoch, num_channels))

    return averaged_epoch_data

def reject_epochs_from_list(epoch_list, samples_per_epoch):
    """
    Removes any epochs from the list that do not have the ideal
    number of samples indicated by samples_per_epoch
    """
    for epoch in epoch_list:
        num_samples = np.shape(epoch.sample_data)[0]
        if num_samples != samples_per_epoch:
            epoch_list.remove(epoch)

    return epoch_list

def reject_epochs_from_matrix(epoch_matrix, samples_per_epoch):
    """
    Removes any epochs from the matrix that do
    not have the ideal number of samples given by samples_per_epoch
    """
    processed_matrix = []

    for epoch_list in epoch_matrix:
        processed_matrix.append(reject_epochs_from_list(epoch_list, samples_per_epoch=samples_per_epoch)) 

    return processed_matrix

def output_epoch_matrix(epoch_matrix, directory):
    """
    Outputs a matrix of epochs to csv files
    """
    for matrix_row in epoch_matrix:
        for epoch_index in range(len(matrix_row)):
            epoch = matrix_row[epoch_index]
            epoch.output_to_csv(epoch_index, directory=directory)


class Epoch(object):

    """Manages segments of EEG epoch data"""
    def __init__(self, is_row, index, start_time, is_p300=False):
        self.is_row = is_row            # Boolean indicating if this epoch is for a row or column
        self.index = index              # Row/Col index of this epoch
        self.start_time = start_time    # Starting time of this epoch
        self.sample_data = None         # Numpy 2D array of sample data (sample X channel)
        self.is_p300 = is_p300      # Set to true if there is supposed to be P300 (used in training)

    def is_within_epoch(self, sample_time, epoch_length):
        """
        Returns true if the given time is within this given epoch
        """
        return (sample_time >= self.start_time
                and sample_time <= self.start_time + epoch_length)
    
    def get_event_code(self):
        """
        Returns an event code representation of this epoch
        """
        orientation = ''
        if self.is_row:
            orientation = 'row'
        else:
            orientation = 'col'
        return encode_event(orientation, self.index, self.is_p300)
    
    def create_epoch_from_code(event_code, start_time):
        """
        Returns a new instance of an epoch given an
        event code and start time
        """
        orientation, index, is_p300 = decode_event(event_code)
        return Epoch(orientation == 'row', index, start_time, is_p300=is_p300)
    create_epoch_from_code = staticmethod(create_epoch_from_code)
    

    def output_to_csv(self, epoch_num, directory, delim=","):
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

    def get_epoch_data(self, data_hist, samples_per_epoch):
        """Gets all the data for this epoch from a given array of sample data history"""
        index_of_first_sample = -1

        for sample_index in range(len(data_hist)):
            if data_hist[sample_index, 0] <= self.start_time:
                index_of_first_sample = sample_index
            elif data_hist[sample_index, 0] > self.start_time:
                if index_of_first_sample == -1:
                    index_of_first_sample = sample_index
                    break

        self.sample_data = np.array(data_hist[
            index_of_first_sample:index_of_first_sample + samples_per_epoch, 1:])

