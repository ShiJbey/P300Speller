# List of tasks for the P300 speller

Deadline: August 9th, 2017

## Tasks
* Command line interface to differentiate when we are: training, gui-only, live,
* Changes to the config: default path for the SVM pickle file
* Imoport a module for classification

## Wednesday, July 12, 2017

The gui works separately from data collection. I still need to get the classifier
set up so that the data can be interpreted. Following that. I need to add the 
ability for the classifier to update the gui with the new value to be added to
the spelling buffer. I think that the spelling buffer needs to be a managed
variable or the speller_gui itself whould be a shared variable that I can call
an "update-spelling-buffer()" method and pass the character that is to be 
added to the buffer. Following that I need to make some GUI changes to help
training run smoother.