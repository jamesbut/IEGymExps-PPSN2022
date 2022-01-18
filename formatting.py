import sys
import pandas as pd

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


# Delete last lines in terminal
def delete_last_lines(n=1):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)


# Formats 2d array into table
def format_data_table(data, row_labels, column_labels, row_axis=None, column_axis=None,
                      show_all_columns=False, dump_file_path=None):

    if show_all_columns:
        pd.set_option('display.max_columns', len(column_labels))

    df = pd.DataFrame(data, index=row_labels, columns=column_labels)

    df = df.rename_axis(row_axis)
    df = df.rename_axis(column_axis, axis='columns')

    print('\n\n')
    print('######################################')
    print('#          Train test table          #')
    print('######################################')
    print('\n\n')
    print(df)

    # Dump to csv file
    if dump_file_path:
        df.to_csv(dump_file_path)


# Formats list to string
def list_to_string(lst):

    lst = [str(i) for i in lst]
    lst = '[' + ' '.join(lst) + ']'

    return lst
