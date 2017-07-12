function x = graphAllEpochs(direction, index, plot_average, plot_epochs)

% direction : Only accepts "row or col"
% index : Value [0,4] indicating the row or column
% plot_average: Should we plot the average of the sample data
% plot_epochs: Should we plot the individual epochs
% [DEPR]epoch_start: What epoch do we want to start with
% [DEPR]epoch_end: What epoch do we want to end with
file_name_root = "";

if direction == "row" || direction == "Row"
    file_name_root = file_name_root + "Row_" + index;
else
    file_name_root = file_name_root + "Col_" + index;
end
file_name_root = file_name_root + "_epoch_*.csv";
file_names = dir(char(file_name_root));

file_names = {file_names.name};
nfiles = length(file_names);

sample_data = cell(1,nfiles);

for index = 1:nfiles
   sample_data{index} = csvread(char(file_names{index}));
end

average = zeros(250,4);

for index = 1:nfiles
   %disp("Iteration: " + index);
   for row = 2:250
       if row > length(sample_data{index})
           break;
       end
       %disp("MatrixA: " + size(average(row,1:4)));
       %disp("MatrixB: " + size(sample_data{index}(row,1:4)));
       average(row,1:4) = average(row,1:4) + sample_data{index}(row,1:4);
   end
end

average =  average / nfiles;

if plot_average
    plot(average(:,1));
end

hold on
if plot_epochs
    for index = 1:nfiles
        plot(sample_data{index}(:,1));
    end
end
hold off
end