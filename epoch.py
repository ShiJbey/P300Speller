import numpy as np
import config


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
                                        index_of_first_sample:index_of_first_sample + SAMPLES_PER_EPOCH,1:],
                                    dtype=np.float64)