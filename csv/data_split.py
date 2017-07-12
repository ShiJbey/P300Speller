import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


file_names = []

if len(sys.argv) > 1:
    # Get all the files
    file_names = sys.argv[1:]

for name in file_names:

    combined_file = open(dir_path + "/" + name, 'r')
    serial_file = open(dir_path + "/serial_test/" + name, 'w')
    rand_file = open(dir_path + "/rand_test/" + name, 'w')

    # Each epoch has its own start time
    start_times_found = 0

    line = combined_file.readline()
    line_count = 0
    while line != "":
        if line.find(",") == -1:
            # We have found a start time
            start_times_found += 1
        # Write to proper file
        if start_times_found == 1:
            serial_file.write(line)
        elif start_times_found == 2:
            rand_file.write(line)
        line = combined_file.readline()
    
    combined_file.close()
    serial_file.close()
    rand_file.close()

