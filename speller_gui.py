# speller_gui.py

import os
from p300_speller import EEGEpoch
import timeit
import time
import Tkinter as tk

class SelectionRectangle():
    """
    Class to maintain information related to the rectangle
    used to select chaacters on the screen
    """

    def __init__(self, root, x=0, y=0, length=10, width=10, color="#ffffff", max_x=100, max_y=100):
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.graphic_ref = None
        self.color = color
        self.max_x = max_x
        self.max_y = max_y

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
        return self.length > self.width
    
    def update(self, row_data, col_data):
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

        if self.is_vertical():
            #print("Current Col: %d" %(int(self.x / self.width)))
            epoch = EEGEpoch(timeit.default_timer())
            epochs = col_data[int(self.x / self.width)]
            epochs["active"].append(epoch)
            col_data[int(self.x / self.width)] = epochs
        else:
            #print("Current Row %d" %(int(self.y / self.length)))
            epoch = EEGEpoch(timeit.default_timer())
            epochs = row_data[int(self.y / self.length)]
            epochs["active"].append(epoch)
            row_data[int(self.y / self.length)] = epochs
        time.sleep(.08)

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
        

class SuggestedWord(tk.Label):
    """
    Responsible for presenting suggested words
    from Qcompleter.
    
    It has a frquency for flashing on and off
    """
    def __init__(self, master, root, text="", frequency=1):
        tk.Label.__init__(self, master, text=text)
        self.frequency = frequency
        self.visible = True

    def update(self):
        if self.visible:
            self["fg"] = '#000000'
        else:
            self["fg"] = '#ffffff'
        self.visible = not self.visible
        #self.root.after(1000/self.frequency, self.update)

    
class P300GUI(tk.Frame):

    def __init__(self, master, row_data, col_data, update_rate=100):
        tk.Frame.__init__(self, master)
        self.row_data = row_data
        self.col_data = col_data
        self.update_rate = update_rate
        self.master["bg"] = '#001c33'
        self["bg"] = '#001c33'
        self.suggested_words = [None,None,None]
        self.grid(padx=30,pady=30)
        self.create_widgets()
        self.set_rectangle()
        self.draw()
        self.update()

    def update(self):
        self.selection_rect.draw(self.canvas)
        self.selection_rect.update(self.row_data, self.col_data)
        self.master.after(self.update_rate, self.update)
        

    def draw(self):
        self.selection_rect.draw(self.canvas)

    def set_rectangle(self, orientation="vertical"):
        col_width = int(self.canvas["width"]) / 6
        self.selection_rect = SelectionRectangle(root=self.master,
                                                x=0,
                                                y=0,
                                                width=col_width,
                                                length=int(self.canvas["height"]),
                                                color="#69778c",
                                                max_x=int(self.canvas["width"]),
                                                max_y=int(self.canvas["height"]))

    def draw_characters(self):
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
                        (row_height * row) + (row_height / 2), anchor="nw")
                self.canvas.itemconfig(canvas_id, text=cell_char, fill="#ffffff")

    def create_widgets(self):
        self.text_buffer = tk.Label(self, text="Words being spelled..")
        self.text_buffer.grid(row=0,sticky=tk.W+tk.E)
        self.text_buffer["fg"] = '#ffffff'
        self.text_buffer["bg"] = '#000000'
        self.text_buffer["pady"] = 30
        
        self.suggested_word_pane = tk.Frame(self)
        self.suggested_word_pane.grid(pady=10)
        
        self.suggested_words[0] = SuggestedWord(self.suggested_word_pane, root=self.master, text="WORD1")
        self.suggested_words[0]["fg"] = '#ffffff'
        self.suggested_words[0]["bg"] = '#000000'
        self.suggested_words[0]["padx"] = 30
        self.suggested_words[0].grid(row=1,column=0)
        
        self.suggested_words[1] = SuggestedWord(self.suggested_word_pane, root=self.master, text="word2")
        self.suggested_words[1]["fg"] = '#ffffff'
        self.suggested_words[1]["bg"] = '#000000'
        self.suggested_words[1]["padx"] = 30
        self.suggested_words[1].grid(row=1,column=1)

        self.suggested_words[2] = SuggestedWord(self.suggested_word_pane, root=self.master, text="word3")
        self.suggested_words[2]["fg"] = '#ffffff'
        self.suggested_words[2]["bg"] = '#000000'
        self.suggested_words[2]["padx"] = 30
        self.suggested_words[2].grid(row=1,column=2)

        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=2,sticky=tk.W+tk.E)
        self.canvas["height"] =  self.canvas["width"]
        self.canvas["bg"] = '#000000'
        self.draw_characters()
          
        self.bottom_button_pane = tk.Frame(self)
        self.bottom_button_pane.grid(pady=10)

        self.back_space_button = tk.Button(self.bottom_button_pane, text='delete', height=1, width=6)
        self.back_space_button.grid(row=0,column=0)

        self.space_button = tk.Button(self.bottom_button_pane, text='space', height=1, width=12)
        self.space_button.grid(row=0,column=1)

        self.send_button = tk.Button(self.bottom_button_pane, text='send', height=1, width=6)
        self.send_button.grid(row=0,column=2)

        self.exit_button = tk.Button(self.bottom_button_pane, text='exit', command=self.master.quit, height=1, width=6)
        self.exit_button.grid(row=0,column=3)
       
