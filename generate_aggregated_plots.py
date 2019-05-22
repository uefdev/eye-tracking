#!/usr/bin/env python3

import csv
import re
import numpy as np
from argparse import ArgumentParser
from math import sqrt
from functools import reduce


def read_csv(csv_filename):
    with open(csv_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        lines = [row for row in csv_reader]
        # Labels, user_ids, data
        return (
            lines[0][1:],
            list(np.array(lines[1:])[:, 0]),
            np.array(lines[1:])[:, 1:].astype(float)
        )


def get_std_of_stds(vector):
    return sqrt(reduce(lambda acc, cur: acc + cur ** 2, vector) / len(vector))


def choose_operation_by_label(labels):
    # True = std, False = mean
    return list(map(lambda label: True if re.search("SD", label) else False, labels))


def main():
    parser = ArgumentParser(
        description="Aggregated value generator"
    )
    parser.add_argument(
        "--input-filename",
        help="Input CSV filename",
        type=str,
        dest="input_filename",
        default="group8_fixations.csv"
    )
    args = parser.parse_args()

    labels, user_ids, data = read_csv(args.input_filename)

    values_for_plotting = np.array(list(map(
        lambda pair: (
            pair[0],
            get_std_of_stds(pair[1]) if choose_operation_by_label(pair[0]) else np.mean(pair[1])
        ),
        zip(labels, data.T)
    )))

    print(values_for_plotting)


if __name__ == "__main__":
    main()
