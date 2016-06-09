#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd

def main(input_file, student_id, delimiter, seed=293, kept_percentage=0.8):
    """Split data rows based on student it and write splits to file.

    :param str input_file: input file name
    :param str student_id: column identifying students
    :param str delimiter: row entries delimiter
    :param int seed: seed for the random split
    :param float kept_percentage: percent of students to retain in first split
    """

    # parse delimiter special characters
    delimiter = delimiter.decode('string_escape')
    df = pd.read_csv(input_file, delimiter=delimiter, index_col=False)
    user_ids = df[student_id].unique()
    np.random.seed(seed)
    np.random.shuffle(user_ids)
    kept_user_ids = user_ids[:int(len(user_ids) * kept_percentage)]
    kept_df = df[df[student_id].isin(kept_user_ids)]
    not_kept_df = df[~df[student_id].isin(kept_user_ids)]

    kept_df.to_csv('.'.join(input_file.split('.')[:-1]) + '_big.txt', index=False, sep=delimiter)
    not_kept_df.to_csv('.'.join(input_file.split('.')[:-1]) + '_small.txt', index=False, sep=delimiter)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "split_data.py filename id_column_name delimiter"
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
