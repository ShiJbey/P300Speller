"""
The idea for these event codes is that they are 4 bit codes.
where:
 - The most significant bit indicates if this epoch contains p300 data
 - The following bit indicates row or column 0->row & 1->Column
 - Following 3 bits indicate the index of the row or column [0,5]
"""

# Non-p300 row event codes
ROW0 = 0
ROW1 = 1
ROW2 = 2
ROW3 = 3
ROW4 = 4
ROW5 = 5

# Non-p300 column event codes
COL0 = 8
COL1 = 9
COL2 = 10
COL3 = 11
COL4 = 12
COL5 = 13

# p300 row event codes
P300_ROW0 = 16
P300_ROW1 = 17
P300_ROW2 = 18
P300_ROW3 = 19
P300_ROW4 = 20
P300_ROW5 = 21

# p300 column event codes
P300_COL0 = 24
P300_COL1 = 25
P300_COL2 = 26
P300_COL3 = 27
P300_COL4 = 28
P300_COL5 = 29

# Masks used in analyzing bits
INDEX_MASK = 7
ORIENTATION_MASK = 8
P300_MASK = 16

class InvalidEventCodeException(Exception):
    """Raised whenever an invalid event code is used"""
    pass

def decode_event(event_code):
    """
    Given an event code, returns a tuple indicating 'row' or 'col'
    and the coresponding index
    """

    index = get_code_index(event_code)
    orientation = get_code_orientation(event_code)
    is_p300 = is_code_p300(event_code)

    if index >= 0 and index <= 5:
        if orientation == 1:
            return 'col', index, is_p300
        else:
            return 'row', index, is_p300
    else:
        raise InvalidEventCodeException()

def encode_event(orientation, index, is_p300=False):
    """
    Given an orientation ('row' or 'col') and an index
    returns the coresponding event code
    """
    if index < 0 or index > 5:
        print "Something is wriong with index"
        raise InvalidEventCodeException()

    if orientation == 'row':
        return (int(is_p300) << 4) + (1 << 3) + index
    elif orientation == 'col':
        return (int(is_p300) << 4) + index
    else:
        print "Something is wriong with orientation"
        raise InvalidEventCodeException()
    

def get_code_index(event_code):
    """
    Retuns the ondex for this code
    """
    return event_code & INDEX_MASK

def get_code_orientation(event_code):
    """
    Returns if this code is for a row or column
    """
    if (event_code & ORIENTATION_MASK) >> 3 == 1:
        return 'col'
    else:
        return 'row'

def is_code_p300(event_code):
    """
    Returns true if this code is a P300 code
    """
    return (event_code & P300_MASK) >> 4 == 1
