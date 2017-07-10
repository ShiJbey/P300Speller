import os
from p300_speller import EEGEpoch
import timeit
import time
import random
import Tkinter as tk
import pylsl
import config

class SelectionRectangle():
    """
    Class to maintain information related to the rectangle
    used to select chaacters on the screen
    """

    def __init__(self, root, x=0, y=0, length=10, width=10,
            color="#ffffff", max_x=100, max_y=100):
        self.root = root
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.graphic_ref = None
        self.color = color
        self.max_x = max_x
        self.max_y = max_y
        self.remaining_rows = []
        self.remaining_cols = []

    def move_to_col(self, index):
        if (self.is_vertical()):
            self.x = index * self.width
        y = 0

    def move_to_row(self, index):
        if (not self.is_vertical()):
            self.y = index * self.length
        x = 0
            
    def rotate90(self):
        """Rotates the rectangle 90 degrees"""
        temp = self.width
        self.width = self.length
        self.length = temp

    def move_vertical(self, distance):
        """Moves the rectangle by some distance in the y-direction"""
        self.y += distance

    def move_horizontal(self, distance):
        """Moves the rectangle by some distance in the x-direction"""
        self.x += distance

    def is_vertical(self):
        """Returns true if the rectangle is oriented vertically"""
        return self.length > self.width
    
    def update(self, row_epoch_queue, col_epoch_queue):
        """Moves the recangle by one row or column and creates epoch"""
        
        if config.RANDOM_HIGHLIGHT: 
            # Takes care of rotating the rectangle when necessary
            if (len(self.remaining_cols) == 0 and self.is_vertical() and
                len(self.remaining_rows) == 0):
                # Make the rectangle horizontal
                self.rotate90()
                self.remaining_rows = range(6)
            elif (len(self.remaining_rows) == 0 and len(self.remaining_cols) == 0 and
                  not self.is_vertical()):
                # Make rect vertical
                self.rotate90()
                self.remaining_cols = range(6)

            # Moves the rectangle to its next position
            if (self.is_vertical()):
                self.y = 0
                randIndex = random.randint(0, len(self.remaining_cols) - 1)
                nextCol = self.remaining_cols[randIndex]
                self.move_to_col(nextCol)
                self.remaining_cols.remove(nextCol)
            else:
                self.x = 0
                randIndex = random.randint(0, len(self.remaining_rows) - 1)
                nextRow = self.remaining_rows[randIndex]
                self.move_to_row(nextRow)
                self.remaining_rows.remove(nextRow)

        else:
            # Move the rectangle
            if self.is_vertical():
                self.move_horizontal(self.width)
                if self.x + self.width > self.max_x:
                    self.x = 0
                    self.rotate90()
            else:
                self.move_vertical(self.length)
                if self.y + self.length > self.max_y:
                    self.y = 0
                    self.rotate90()
                    
        # Create a new epoch for this update
        self.send_epoch(row_epoch_queue, col_epoch_queue)
        
    def get_index(self):
        if (self.is_vertical()):
            return int(self.x / self.width)
        else:
            return int(self.y / self.length)

    def send_epoch(self, row_epoch_queue, col_epoch_queue):
        # Send a new epoch to the main process
        if self.is_vertical():
            #print("Current Col: %d" %(int(self.x / self.width)))
            index = int(self.x / self.width)
            epoch = EEGEpoch(False, index, pylsl.local_clock())
            col_epoch_queue.put_nowait(epoch)
        else:
            #print("Current Row %d" %(int(self.y / self.length)))
            index = int(self.y / self.length)
            epoch = EEGEpoch(True, index, pylsl.local_clock())
            row_epoch_queue.put_nowait(epoch)
    
    def makeInvisible(self, canvas):
        canvas.delete(self.graphic_ref)

    def draw(self, canvas):
        """Draws the rectange to a Tkinter canvas"""
        # Dispose of old drawn rectangle
        canvas.delete(self.graphic_ref)
        # Draw new rectangle and save reference
        self.graphic_ref = canvas.create_rectangle(self.x,
                                                  self.y,
                                                  self.x + self.width,
                                                  self.y + self.length,
                                                  fill=self.color)


class StartScreen(tk.Frame):
    """This class just serves as the starting screen for the application"""
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        self.title_text = tk.Label(self, text="P300 Speller (WIP)", font=("Arial", 24))
        self.title_text.grid(row=0,)

        self.start_button = tk.Button(self, command=self.start_speller, text='Start', font=("Arial", 24, "bold"), height=4, width=24)
        self.start_button.grid(row=2,pady=3, sticky=tk.W+tk.E)

        self.grid(padx=30,pady=30)

    def start_speller(self):
        """Removes this frame and displays the grid of characters"""
        self.master.create_widgets()
        self.master.set_rectangle()
        self.master.draw()
        self.master.update()
        self.grid_remove()
        

    
class P300GUI(tk.Frame):

    def __init__(self, master, row_epoch_queue, col_epoch_queue, highlight_time=.100, intermediate_time=80):
        tk.Frame.__init__(self, master)
        self.row_epoch_queue = row_epoch_queue
        self.col_epoch_queue = col_epoch_queue
        self.highlight_time = highlight_time
        self.intermediate_time = intermediate_time
        self.rect_visible = True
        self.master["bg"] = '#001c33'
        self["bg"] = '#001c33'
        self.selection_rect = None
        self.canvas = tk.Canvas(self)
        self.canvas["width"] = 500
        self.display_start_screen()
        self.grid(padx=30,pady=30)

    def display_start_screen(self):
        start_screen = StartScreen(self)

    def update(self):
        """Updates the position and visibility of the selection rectangle""" 
        if (self.rect_visible):
            # Make the rectangle invisible after it has been presented
            self.selection_rect.makeInvisible(self.canvas)
            self.draw()
            self.rect_visible = not self.rect_visible
            self.master.after(self.highlight_time, self.update)
        else:
            # Make the rectangle visible for a given amount of time
            self.selection_rect.update(self.row_epoch_queue, self.col_epoch_queue)
            self.draw()
            self.rect_visible = not self.rect_visible
            self.master.after(self.intermediate_time, self.update)
        
    def draw(self):
        self.canvas.delete("all")
        if (self.rect_visible):
            self.selection_rect.draw(self.canvas)
        self.draw_characters()

    def set_rectangle(self, orientation="vertical"):
        """Creates a new selection rectangle for this gui"""
        col_width = int(self.canvas["width"]) / 6
        self.selection_rect = SelectionRectangle(root=self.master,
                                                x=0,
                                                y=0,
                                                width=col_width,
                                                length=int(self.canvas["height"]),
                                                color="#ffffff",
                                                max_x=int(self.canvas["width"]),
                                                max_y=int(self.canvas["height"]))

    def draw_characters(self):
        """Draws 36 characters [a-z] and [0-9] in a 6x6 grid"""
        row_height = int(self.canvas["height"]) / 6
        col_width = int(self.canvas["width"]) / 6
        
        # Draw the characters to the canvas
        ascii_letter_offset = 65
        ascii_number_offset = 48
        ascii_offset = ascii_letter_offset
        current_offset = 0

        

        for row in range(6):
            for col in range(6):
                # Case that we have gone through all characters
                if current_offset == 26 and ascii_offset == ascii_letter_offset:
                    ascii_offset = ascii_number_offset
                    current_offset = 0
                # Case that we have gone though all of the numbers
                elif current_offset == 10 and ascii_offset == ascii_number_offset:
                    break
                # Get the current cell character
                cell_char = chr(ascii_offset + current_offset)
                current_offset += 1
                canvas_id = self.canvas.create_text((col_width * col) + (col_width / 2),
                        (row_height * row) + (row_height / 4), font=("Arial", (col_width / 4),"bold"), anchor="nw")
                
                # Determine if this character is printed white or black
                if (self.selection_rect != None):
                    if ((self.selection_rect.is_vertical() and col == self.selection_rect.get_index()
                        or not self.selection_rect.is_vertical() and row == self.selection_rect.get_index())
                        and self.rect_visible):
                        self.canvas.itemconfig(canvas_id, text=cell_char, fill="#000000")
                    else:
                        self.canvas.itemconfig(canvas_id, text=cell_char, fill="#ffffff")

    def create_widgets(self):
        """Populates the gui with all the necessary components"""
        # Displays the current text being typed
        self.text_buffer = tk.Label(self, text="")
        self.text_buffer.grid(row=0, sticky=tk.W+tk.E)
        self.text_buffer["fg"] = '#ffffff'
        self.text_buffer["bg"] = '#000000'
        self.text_buffer["pady"] = 30
        # Canvas for drawing the grid of characters and the rectangle
        self.canvas.grid(row=2, sticky=tk.W+tk.E)
        self.canvas["height"] = self.canvas["width"]
        self.canvas["bg"] = '#000000'
        self.draw_characters()
        # Frame to hold all buttons at the bttom of the gui
        self.bottom_button_pane = tk.Frame(self)
        self.bottom_button_pane.grid(pady=10)
        # Button to delete the previous character
        #self.back_space_button = tk.Button(self.bottom_button_pane, text='delete', height=1, width=6)
        #self.back_space_button.grid(row=0,column=0)
        # Button for adding a space character to the text_buffer
        #self.space_button = tk.Button(self.bottom_button_pane, text='space', height=1, width=12)
        #self.space_button.grid(row=0,column=1)
        # Button for exiting the application
        self.exit_button = tk.Button(self.bottom_button_pane, text='exit', command=self.master.quit, height=1, width=6)
        self.exit_button.grid(row=0,column=3)
       
