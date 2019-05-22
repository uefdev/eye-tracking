#!/usr/bin/env python3

import csv
import re
import numpy as np
from argparse import ArgumentParser
from math import sqrt
from functools import reduce
from matplotlib import pyplot as plt


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

    vals = dict(
        (key, value) for key, value in
        map(
            lambda pair: (
                pair[0],
                get_std_of_stds(pair[1]) if choose_operation_by_label(pair[0]) else np.mean(pair[1])
            ),
            zip(labels, data.T)
        )
    )

    mfd_mean = (vals["MFD_true"], vals["MFD_false"], vals["MFD_overall"])
    mfd_sd = (vals["MFD_SD_true"], vals["MFD_SD_false"], vals["MFD_overall_SD"])

    msa_mean = (vals["MSA_true"], vals["MSA_false"], vals["MSA_overall"])
    msa_sd = (vals["MSA_SD_true"], vals["MSA_SD_false"], vals["MSA_overall_SD"])

    ind = np.array(range(3))

    labels = ["known", "unknown", "overall"]
    colors = ["tab:green", "tab:red", "tab:blue"]

    ax = plt.subplot(1, 2, 1)
    ax.bar(ind, mfd_mean, 0.8, yerr=mfd_sd, tick_label=labels, color=colors)
    ax.title.set_text("Mean Fixation Duration (ms)")

    ax = plt.subplot(1, 2, 2)
    ax.bar(ind, msa_mean, 0.8, yerr=msa_sd, tick_label=labels, color=colors)
    ax.title.set_text("Mean Saccade Amplitude")

    plt.show()

    print(vals)


if __name__ == "__main__":
    main()
