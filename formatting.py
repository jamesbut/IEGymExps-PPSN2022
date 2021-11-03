import sys
from pandas import DataFrame

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


# Delete last lines in terminal
def delete_last_lines(n=1):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


# Formats 2d array into table
def format_data_table(data, row_labels, column_labels, row_axis=None, column_axis=None):

    df = DataFrame(data, index=row_labels, columns=column_labels)

    df = df.rename_axis(row_axis)
    df = df.rename_axis(column_axis, axis='columns')

    print('\n\n')
    print('######################################')
    print('#          Train test table          #')
    print('######################################')
    print('\n\n')
    print(df)


# Formats list to string
def list_to_string(lst):

    lst = [str(i) for i in lst]
    lst = '[' + ' '.join(lst) + ']'

    return lst
