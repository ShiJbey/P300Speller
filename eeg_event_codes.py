"""
The idea for these event codes is that they are 5 bit codes.
where:
 - The most significant bit indicates if this epoch contains p300 data
 - The following bit indicates row or column 0->row & 1->Column
 - Following 3 bits indicate the index of the row or column [1,5]
"""

# Non-p300 row event codes
ROW0 = 1
ROW1 = 2
ROW2 = 3
ROW3 = 4
ROW4 = 5
ROW5 = 6

# Non-p300 column event codes
COL0 = 9
COL1 = 10
COL2 = 11
COL3 = 12
COL4 = 13
COL5 = 14

# p300 row event codes
P300_ROW0 = 17
P300_ROW1 = 18
P300_ROW2 = 19
P300_ROW3 = 20
P300_ROW4 = 21
P300_ROW5 = 22

# p300 column event codes
P300_COL0 = 25
P300_COL1 = 26
P300_COL2 = 27
P300_COL3 = 28
P300_COL4 = 29
P300_COL5 = 30

# Code when no event was triggered
NO_EVENT = 0

# Masks used in analyzing bits
INDEX_MASK = 7
ORIENTATION_MASK = 8
P300_MASK = 16

LIVE_EVENT_ID = {
    'nontarget/row0':ROW0,
    'nontarget/row1':ROW1,
    'nontarget/row2':ROW2,
    'nontarget/row3':ROW3,
    'nontarget/row4':ROW4,
    'nontarget/row5':ROW5,
    'nontarget/col0':COL0,
    'nontarget/col1':COL1,
    'nontarget/col2':COL2,
    'nontarget/col3':COL3,
    'nontarget/col4':COL4,
    'nontarget/col5':COL5,
}

EVENT_ID = {
    'nontarget/row0':ROW0,
    'nontarget/row1':ROW1,
    'nontarget/row2':ROW2,
    'nontarget/row3':ROW3,
    'nontarget/row4':ROW4,
    'nontarget/row5':ROW5,
    'nontarget/col0':COL0,
    'nontarget/col1':COL1,
    'nontarget/col2':COL2,
    'nontarget/col3':COL3,
    'nontarget/col4':COL4,
    'nontarget/col5':COL5,
    'target/row0':P300_ROW0,
    'target/row1':P300_ROW1,
    'target/row2':P300_ROW2,
    'target/row3':P300_ROW3,
    'target/row4':P300_ROW4,
    'target/row5':P300_ROW5,
    'target/col0':P300_COL0,
    'target/col1':P300_COL1,
    'target/col2':P300_COL2,
    'target/col3':P300_COL3,
    'target/col4':P300_COL4,
    'target/col5':P300_COL5,
    }

def get_event_tag(event_code):
    for tag in EVENT_ID.keys():
        if EVENT_ID[tag] == event_code:
            return tag
    return 'none'

def get_event_id_dict(p300_row_index, p300_col_index):
    """
    Given indices for the row and column of the desired character
    for this trial, outputs the list of event codes that should have
    been encountered
    """
    event_ids = {}
    p300_row_code = encode_event('row', p300_row_index, is_p300=True)
    p300_col_code = encode_event('col', p300_col_index, is_p300=True)

    for row_index in range(6):
        if row_index == p300_row_index:
            # Add the row target event code
            event_tag = get_event_tag(p300_row_code)
            event_ids[event_tag] = p300_row_code
        else:
            # add the non target row code
            event_code = encode_event('row', row_index, is_p300=False)
            event_tag = get_event_tag(event_code)
            event_ids[event_tag] = event_code

    for col_index in range(6):
        if col_index == p300_col_index:
            # Add the row target event code
            event_tag = get_event_tag(p300_col_code)
            event_ids[event_tag] = p300_col_code
        else:
            # add the non target row code
            event_code = encode_event('col', col_index, is_p300=False)
            event_tag = get_event_tag(event_code)
            event_ids[event_tag] = event_code

    return event_ids



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

    if orientation == 'col':
        #returns 'x1xxx'
        return (int(is_p300) << 4) + (1 << 3) + (index + 1)
    elif orientation == 'row':
        #returns 'x0xxx'
        return (int(is_p300) << 4) + (index + 1)
    else:
        print "Something is wriong with orientation"
        raise InvalidEventCodeException()


def get_code_index(event_code):
    """
    Retuns the ondex for this code
    """
    return (int(event_code) & INDEX_MASK) - 1

def get_code_orientation(event_code):
    """
    Returns if this code is for a row or column
    """
    if ((int(event_code) & ORIENTATION_MASK) >> 3) == 1:
        return 'col'
    else:
        return 'row'

def is_code_p300(event_code):
    """
    Returns true if this code is a P300 code
    """
    return ((int(event_code) & P300_MASK) >> 4) == 1
