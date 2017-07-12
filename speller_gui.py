import time
import random
import Tkinter as tk
import pylsl
from p300_speller import EEGEpoch
import config

class SelectionRectangle():
    """Manages the rectangle that highlights the characters in the grid"""
    def __init__(self, x, y, length, width, max_x, max_y, color="#ffffff"):
        self.x = x                  # X-position of the rectangle (top-left corner)
        self.y = y                  # Y-position of the rectangle (top-left corner)
        self.length = length        # How long, in the y-direction is the rectangle
        self.width = width          # How long, in the x-direction is the rectangle
        self.graphic_ref = None     # Reference to the graphical rect on the canvas 
        self.color = color          # Color of the rectangle
        self.max_x = max_x          # Max x-position that the rectangle may occupy
        self.max_y = max_y          # Max y-position that the rectangle may occupy
        self.remaining_rows = range(6)  # List of possible rows for the rect to move to
        self.remaining_cols = range(6)  # List of possible columns for the rect to move to
        self.visible = True         # Is the rectangle currently made visible on the screen 

    def move_to_col(self, index):
        """Moves and re-orients the rectangle to a column specified by an index"""
        # Reorient the rectangle to be vertical
        if not self.is_vertical():
            self.rotate90()
        # Set the rectangle to the proper position    
        self.x = index * self.width
        self.y = 0

    def move_to_row(self, index):
        """Moves and re-orients the rectangle to a row specified by an index"""
        # Reorient the rectangle to be horizontal 
        if self.is_vertical():
            self.rotate90()
        # Set the rectangel to the proper position
        self.y = index * self.length
        self.x = 0
            
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

    def refill_available_rcs(self):
        """Refills the lists of available rows and columns with index values"""
        self.remaining_rows = range(6)
        self.remaining_cols = range(6)
    
    def select_rand_row(self):
        """Selects a row from the available_rows"""
        rand_index = random.randint(0, len(self.remaining_rows) - 1)
        row = self.remaining_rows[rand_index]
        return row
    
    def select_rand_col(self):
        """Selects a random column from the available_cols"""
        rand_index = random.randint(0, len(self.remaining_cols) - 1)
        col = self.remaining_cols[rand_index]
        return col
    
    def end_of_sequence(self):
        """Returns true if there are no more available moves for the rect"""
        return len(self.remaining_cols) == 0 and len(self.remaining_rows) == 0

    def update(self, row_epoch_queue, col_epoch_queue):
        """Moves the recangle by one row or column and creates epoch"""
        # Move the rectangle by randomly selecting a row or column
        if config.RANDOM_HIGHLIGHT:
            # The remaining columns and row lists need to be refilled
            if self.end_of_sequence():
                self.refill_available_rcs()

            # Freely choose between available rows and columns
            if len(self.remaining_cols) > 0 and len(self.remaining_rows) > 0:
                if random.random() > 0.5:
                    next_col = self.select_rand_col()
                    self.move_to_col(next_col)
                    self.remaining_cols.remove(next_col)
                else:
                    next_row = self.select_rand_row()
                    self.move_to_row(next_row)
                    self.remaining_rows.remove(next_row)

            # Only rows are available to chose from
            elif len(self.remaining_cols) == 0:
                next_row = self.select_rand_row()
                self.move_to_row(next_row)
                self.remaining_rows.remove(next_row)

            # Only columns are available to chose from
            elif len(self.remaining_rows) == 0:
                next_col = self.select_rand_col()
                self.move_to_col(next_col)
                self.remaining_cols.remove(next_col)

        # Move linearly through all the rows and columns
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
        """Return the current row or column index of the rectangle"""
        if self.is_vertical():
            return int(self.x / self.width)
        else:
            return int(self.y / self.length)

    def send_epoch(self, row_epoch_queue, col_epoch_queue):
        """Send a new epoch to the main process"""
        if self.is_vertical():
            index = self.get_index()
            epoch = EEGEpoch(False, index, pylsl.local_clock())
            col_epoch_queue.put_nowait(epoch)
        else:
            index = self.get_index()
            epoch = EEGEpoch(True, index, pylsl.local_clock())
            row_epoch_queue.put_nowait(epoch)
    
    def draw(self, canvas):
        """Draws the rectange to a Tkinter canvas"""
        if self.visible:
            # Dispose of old drawn rectangle
            if self.graphic_ref != None:
                canvas.delete(self.graphic_ref)
            # Draw new rectangle and save reference
            self.graphic_ref = canvas.create_rectangle(self.x,
                                                        self.y,
                                                        self.x + self.width,
                                                        self.y + self.length,
                                                        fill=self.color)


class StartScreen(tk.Frame):
    """Starting screen for the application"""
    def __init__(self, master, next_screen):
        tk.Frame.__init__(self, master)
        # Add title text
        self.title_text = tk.Label(self, text="P300 Speller (WIP)", font=("Arial", 24))
        self.title_text.grid(row=0,)
        # Add start button
        self.start_button = tk.Button(self, command=self.start_speller, text='Start', font=("Arial", 24, "bold"), height=4, width=24)
        self.start_button.grid(row=2,pady=3, sticky=tk.W+tk.E)
        # Screen that comes after this screen
        self.next_screen = next_screen
    
    def display_screen(self):
        """Adds this screen to the window"""
        self.grid(padx=30,pady=30)
    
    def remove_screen(self):
        """Removes this screen from the window"""
        self.grid_remove()

    def start_speller(self):
        """Removes this frame and displays the grid of characters"""
        self.next_screen.display_screen()
        self.next_screen.update()
        self.remove_screen()
        
    
class P300GUI(tk.Frame):
    """The main screen of the application that displays the character grid and spelling buffer"""
    def __init__(self, master, row_epoch_queue, col_epoch_queue,
        highlight_time=config.HIGHLIGHT_TIME, intermediate_time=config.INTERMEDIATE_TIME):
        tk.Frame.__init__(self, master)
        self.row_epoch_queue = row_epoch_queue      # Reference to the queue of row epochs shared with main process
        self.col_epoch_queue = col_epoch_queue      # Reference to the queue of col epochs shared with main process
        self.selection_rect = self.make_rectangle()      # Selection rectangle used in the GUI
        self.highlight_time = highlight_time        # How long the highlight rectangle will be present
        self.intermediate_time = intermediate_time  # Length of time between presentations of the highlight rectangle
        self.canvas = tk.Canvas(self)               # Reference to the cavas where the characters and rect are drawn
        self.create_widgets()                       # Populate the screen
        self.master["bg"] = '#001c33'
        self["bg"] = '#001c33'
        
    def display_screen(self):
        """Adds this screen to the window"""
        self.grid(padx=30,pady=30)
    
    def remove_screen(self):
        """Removes this screen from the window"""
        self.grid_remove()

    def update(self):
        """Updates the position and visibility of the selection rectangle""" 
        if self.selection_rect.visible:
            # Update the position of the rectangle
            self.selection_rect.update(self.row_epoch_queue, self.col_epoch_queue)
            # Rectangle is set to visible, draw the canvas
            self.draw()
            # Set it visibility for when this function is called again
            self.selection_rect.visible = False
            # Allow the rectangle to remain visible for a set time
            self.master.after(self.highlight_time, self.update)
        else:
            # Rectangle is set to invisible, update the canvas
            self.draw()
            # Set visibility to visible for next update call
            self.selection_rect.visible = True
            if self.selection_rect.end_of_sequence():
                self.master.after(config.EPOCH_LENGTH * 1000 + self.intermediate_time, self.update)
            else:
                # Keep the rect invisible for a set amount of time
                self.master.after(self.intermediate_time, self.update)
        
    def draw(self):
        """Redraws the canvas"""
        self.canvas.delete("all")
        self.selection_rect.draw(self.canvas)
        self.draw_characters()

    def make_rectangle(self, orientation="vertical"):
        """Returns a new selection rectangle for this GUI"""
        col_width = config.GRID_WIDTH / 6
        if orientation == "vertical":
            return SelectionRectangle(x=0, y=0,
                                    width=col_width,
                                    length=config.GRID_WIDTH,
                                    color=config.RECT_COLOR,
                                    max_x=config.GRID_WIDTH,
                                    max_y=config.GRID_WIDTH)
        else:
            return SelectionRectangle(x=0, y=0,
                                    width=config.GRID_WIDTH,
                                    length=col_width,
                                    color=config.RECT_COLOR,
                                    max_x=config.GRID_WIDTH,
                                    max_y=config.GRID_WIDTH)

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
                        and self.selection_rect.visible):
                        self.canvas.itemconfig(canvas_id, text=cell_char, fill=config.HIGHLIGHT_CHAR_COLOR)
                    else:
                        self.canvas.itemconfig(canvas_id, text=cell_char, fill=config.DEFAULT_CHAR_COLOR)

    def create_widgets(self):
        """Populates the gui with all the necessary components"""
        # Displays the current text being typed
        self.text_buffer = tk.Label(self, text="")
        self.text_buffer.grid(row=0, sticky=tk.W+tk.E)
        self.text_buffer["fg"] = '#ffffff'
        self.text_buffer["bg"] = '#000000'
        self.text_buffer["pady"] = 30
        # Canvas for drawing the grid of characters and the rectangle
        self.canvas["width"] = config.GRID_WIDTH
        self.canvas["height"] = self.canvas["width"]
        self.canvas["bg"] = config.GRID_BG_COLOR
        self.canvas.grid(row=2, sticky=tk.W+tk.E)
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
       