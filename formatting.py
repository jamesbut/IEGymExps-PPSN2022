CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

import sys
from pandas import DataFrame

#Delete last lines in terminal
def delete_last_lines(n=1):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


#Formats 2d array into table
def format_data_table(data, row_labels, column_labels):
    df = DataFrame(data, index=row_labels, columns=column_labels)
    print(df)


#Formats list to string
def list_to_string(l):

    l = [str(i) for i in l]
    l = '[' + ' '.join(l) + ']'

    return l

