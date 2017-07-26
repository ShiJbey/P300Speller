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

class InvalidEventCodeException(Exception):
    """Raised whenever an invalid event code is used"""
    pass

def decode_event(event_code):
    """
    Given an event code, returns a tuple indicating 'row' or 'col'
    and the coresponding index
    """
    index_mask = 7
    index = event_code & index_mask

    if index >= 0 and index <= 5:
        if event_code >> 3 == 1:
            return (event_code >> 4 == 1, 'col', index)
        elif event_code >> 3 == 0:
            return (event_code >> 4 == 1, 'row', index)
        else:
            raise InvalidEventCodeException()
    else:
        raise InvalidEventCodeException()

def get_event_code(orientation, index, is_p300=False):
    """
    Given an orientation ('row' or 'col') and an index
    returns the coresponding event code
    """
    if index < 0 or index > 5:
        raise InvalidEventCodeException()

    if orientation == 'row':
        return (int(is_p300) << 4) + (1 << 3) + index
    elif orientation == 'col':
        return (int(is_p300) << 4) + index
    else:
        raise InvalidEventCodeException()
        