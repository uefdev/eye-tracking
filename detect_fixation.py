#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from functools import reduce
from zipfile import ZipFile

"""
PSEUDOCODE

Inputs:
   Points, dispersion threshold, duration threshold

While points left:
   Set moving window to the first points within duration threshold

    disp = Dispersion of points in the window
    If disp <= threshold:
        Until disp > threshold:
          Add next point to window
          Update disp

        Remove last point from window
        coord = Centroid of all points in window
        Add coord to list of fixations
        Remove points inside window from points list

     Else:
         Remove the first point from points list
Return list of fixations
"""

# Constants
KNOWN = "known"
UNKNOWN = "unknown"

MFD_TRUE = "MFD_true"
MFD_SD_TRUE = "MFD_SD_true"

MFD_FALSE = "MFD_false"
MFD_SD_FALSE = "MFD_SD_false"

MFD_OVERALL = "MFD_overall"
MFD_OVERALL_SD = "MFD_overall_SD"

MSA_TRUE = "MSA_true"
MSA_SD_TRUE = "MSA_SD_true"

MSA_FALSE = "MSA_false"
MSA_SD_FALSE = "MSA_SD_false"

MSA_OVERALL = "MSA_overall"
MSA_OVERALL_SD = "MSA_overall_SD"


def zip_coords(coordinate_vector):
    return list(zip(coordinate_vector[0::2], coordinate_vector[1::2]))


def calculate_mfd(structs):
    return np.mean(list(map(lambda struct: struct[0], reduce(lambda acc, cur: acc + cur[1], structs, []))))


def calculate_mfd_sd(structs):
    return np.std(list(map(lambda struct: struct[0], reduce(lambda acc, cur: acc + cur[1], structs, []))))


def calculate_msa(structs):
    return np.mean(reduce(lambda acc, cur: acc + cur[2], structs, []))


def calculate_msa_sd(structs):
    return np.std(reduce(lambda acc, cur: acc + cur[2], structs, []))


def get_filtered_data_as_dict(archive_name, file_name, target_users, dispersion_threshold, duration_threshold):
    def data_reducer(acc, cur):
        cur_fixation_points = detect_fixation(
            original_points=cur[2],
            dispersion_threshold=dispersion_threshold,
            duration_threshold=duration_threshold
        )
        cur_saccade_amplitudes = calculate_saccade_amplitudes(
            points=cur[2],
            fixation_points=cur_fixation_points
        )
        sample_values = (
            cur[2],
            cur_fixation_points,
            cur_saccade_amplitudes
        )
        acc[cur[0]][KNOWN].append(sample_values) if cur[1] == "true" else acc[cur[0]][UNKNOWN].append(sample_values)
        return acc

    data = reduce(
        data_reducer,
        list(map(
            lambda row: [row[0], row[1], zip_coords(row[2:])],
            map(
                lambda row: [
                    str(value) if index == 0 or index == 1 else float(value) for index, value in enumerate(row)
                ],
                filter(
                    lambda row: row[0] in target_users and len(row) > 1,
                    map(
                        lambda row: row.split(","),
                        ZipFile(archive_name, "r").read(file_name).decode("utf-8").split("\n")
                    )
                )
            )
        )),
        dict({
            user_id: dict({
                MFD_TRUE: None,
                MFD_SD_TRUE: None,
                MFD_FALSE: None,
                MFD_SD_FALSE: None,
                MFD_OVERALL: None,
                MFD_OVERALL_SD: None,
                MSA_TRUE: None,
                MSA_SD_TRUE: None,
                MSA_FALSE: None,
                MSA_SD_FALSE: None,
                MSA_OVERALL: None,
                MSA_OVERALL_SD: None,
                KNOWN: [],
                UNKNOWN: []
            }) for user_id in target_users
        })
    )

    def calculate_metadata(data_dictionary):
        for user_id in data_dictionary:
            # MFD
            data_dictionary[user_id][MFD_TRUE] = calculate_mfd(data_dictionary[user_id][KNOWN])
            data_dictionary[user_id][MFD_SD_TRUE] = calculate_mfd_sd(data_dictionary[user_id][KNOWN])

            data_dictionary[user_id][MFD_FALSE] = calculate_mfd(data_dictionary[user_id][UNKNOWN])
            data_dictionary[user_id][MFD_SD_FALSE] = calculate_mfd_sd(data_dictionary[user_id][UNKNOWN])

            data_dictionary[user_id][MFD_OVERALL] = calculate_mfd(
                data_dictionary[user_id][KNOWN] + data_dictionary[user_id][UNKNOWN]
            )
            data_dictionary[user_id][MFD_OVERALL_SD] = calculate_mfd_sd(
                data_dictionary[user_id][KNOWN] + data_dictionary[user_id][UNKNOWN]
            )

            # MSA
            data_dictionary[user_id][MSA_TRUE] = calculate_msa(data_dictionary[user_id][KNOWN])
            data_dictionary[user_id][MSA_SD_TRUE] = calculate_msa_sd(data_dictionary[user_id][KNOWN])

            data_dictionary[user_id][MSA_FALSE] = calculate_msa(data_dictionary[user_id][UNKNOWN])
            data_dictionary[user_id][MSA_SD_FALSE] = calculate_msa_sd(data_dictionary[user_id][UNKNOWN])

            data_dictionary[user_id][MSA_OVERALL] = calculate_msa(
                data_dictionary[user_id][KNOWN] + data_dictionary[user_id][UNKNOWN]
            )
            data_dictionary[user_id][MSA_OVERALL_SD] = calculate_msa_sd(
                data_dictionary[user_id][KNOWN] + data_dictionary[user_id][UNKNOWN]
            )

        return data_dictionary

    return calculate_metadata(data)


def calculate_dispersion(points):
    x_max = max(points, key=lambda pair: pair[0])[0]
    y_max = max(points, key=lambda pair: pair[1])[1]
    x_min = min(points, key=lambda pair: pair[0])[0]
    y_min = min(points, key=lambda pair: pair[1])[1]
    return (x_max - x_min) + (y_max - y_min)


# Identification by dispersion threshold
def detect_fixation(original_points, dispersion_threshold=80, duration_threshold=100):
    points = original_points.copy()
    fixation_points = []
    removed = 0

    # While there are still points
    while points:

        # Initialize window over first points to cover the duration threshold
        window = []
        for i in range(duration_threshold):
            try:
                window.append(points[i])
            except IndexError:
                break
                # return fixation_points

        # If dispersion of window points <= threshold
        dispersion = calculate_dispersion(window)
        if dispersion <= dispersion_threshold:

            # Add additional points to the window until dispersion > threshold
            i = 0
            while True:
                try:
                    window.append(points[duration_threshold + i])
                    i += 1
                except IndexError:
                    break
                dispersion = calculate_dispersion(window)
                if dispersion > dispersion_threshold:
                    window.pop()
                    break

            # Note a fixation at the centroid of the window points
            centroid = np.mean(window, axis=0)

            # Remove window points from points
            fixation_start = 0
            for i in range(len(window)):
                if i == 0:
                    fixation_start = removed
                points.pop(0)
                removed += 1
            fixation_end = removed - 1
            fixation_points.append(
                (len(window), centroid, fixation_start, fixation_end)
            )

        # Else Remove first point from points
        else:
            points.pop(0)
            removed += 1

    return fixation_points


def calculate_euclidean_distance(start_point, end_point):
    return np.linalg.norm(np.array(start_point - np.array(end_point)))


def calculate_saccade_amplitudes(points, fixation_points):
    saccade_amplitudes = []

    if fixation_points[0][2] != 0:
        distance = calculate_euclidean_distance(points[0], fixation_points[0][1])
        saccade_amplitudes.append(distance)

    if fixation_points[-1][3] != len(points) - 1:
        distance = calculate_euclidean_distance(fixation_points[-1][1], points[-1])
        saccade_amplitudes.append(distance)

    for i in range(len(fixation_points)):
        distance = calculate_euclidean_distance(fixation_points[i][1], fixation_points[i - 1][1])
        saccade_amplitudes.append(distance)

    return saccade_amplitudes


def plot_sample(sample):
    (sample_points, sample_fixation_points, sample_saccade_points) = sample
    sample_fixation_points_array = np.concatenate(np.array(sample_fixation_points)[:, 1]).reshape(-1, 2)
    sample_points_array = np.array(sample_points)
    plt.scatter(sample_points_array[:, 0], sample_points_array[:, 1])
    plt.scatter(sample_fixation_points_array[:, 0], sample_fixation_points_array[:, 1])
    plt.show()


def write_output_csv(output_csv_filename, data_dictionary):
    with open(output_csv_filename, "w") as csv_file:
        file_writer = csv.writer(
            csv_file,
            delimiter=","
        )
        file_writer.writerow([
            "subject_id",
            MFD_TRUE,
            MFD_SD_TRUE,
            MFD_FALSE,
            MFD_SD_FALSE,
            MSA_TRUE,
            MSA_SD_TRUE,
            MSA_FALSE,
            MSA_SD_FALSE,
            MFD_OVERALL,
            MFD_OVERALL_SD,
            MSA_OVERALL,
            MSA_OVERALL_SD
        ])
        for user_id in data_dictionary:
            file_writer.writerow([
                user_id,
                data_dictionary[user_id][MFD_TRUE],
                data_dictionary[user_id][MFD_SD_TRUE],
                data_dictionary[user_id][MFD_FALSE],
                data_dictionary[user_id][MFD_SD_FALSE],
                data_dictionary[user_id][MSA_TRUE],
                data_dictionary[user_id][MSA_SD_TRUE],
                data_dictionary[user_id][MSA_FALSE],
                data_dictionary[user_id][MSA_SD_FALSE],
                data_dictionary[user_id][MFD_OVERALL],
                data_dictionary[user_id][MFD_OVERALL_SD],
                data_dictionary[user_id][MSA_OVERALL],
                data_dictionary[user_id][MSA_OVERALL_SD]
            ])
        csv_file.close()


# subject_id MFD_true MFD_SD_true MFD_false MFD_SD_false MSA_true MSA_SD_true MSA_false MSA_SD_false
# MFD_overall MFD_overall_SD MSA_overall MSA_overall_SD


def main():
    parser = ArgumentParser(
        description="Fixation detection script"
    )
    parser.add_argument(
        "--archive",
        help="ZIP archive filename",
        type=str,
        dest="archive_name",
        default="train.csv.zip"
    )
    parser.add_argument(
        "--input-filename",
        help="Filename inside the ZIP archive",
        type=str,
        dest="input_filename",
        default="train.csv"
    )
    parser.add_argument(
        "--output-filename",
        help="Filename for the CSV file",
        type=str,
        dest="output_filename",
        default="group8_fixations.csv"
    )
    parser.add_argument(
        "--target-users",
        help="List of target user ids",
        type=str,
        dest="target_users",
        nargs="+",
        default=["s8", "s18", "s28", "s4", "s14", "s24"]
    )
    parser.add_argument(
        "--dispersion-threshold",
        help="Dispersion threshold",
        type=int,
        dest="dispersion_threshold",
        default=80
    )
    parser.add_argument(
        "--duration-threshold",
        help="Duration threshold",
        type=int,
        dest="duration_threshold",
        default=100
    )
    parser.add_argument(
        "--show-plots",
        help="Show plots",
        action="store_true",
        dest="show_plots",
    )
    args = parser.parse_args()

    samples = get_filtered_data_as_dict(
        archive_name=args.archive_name,
        file_name=args.input_filename,
        target_users=args.target_users,
        dispersion_threshold=args.dispersion_threshold,
        duration_threshold=args.duration_threshold
    )

    write_output_csv(args.output_filename, samples)

    # Display data
    for user_id in samples:
        print(f"\nUser {user_id}")
        print(f"MFD_TRUE: {samples[user_id][MFD_TRUE]}")
        print(f"MFD_SD_TRUE: {samples[user_id][MFD_SD_TRUE]}")

        print(f"MFD_FALSE: {samples[user_id][MFD_FALSE]}")
        print(f"MFD_SD_FALSE: {samples[user_id][MFD_SD_FALSE]}")

        print(f"MFD_OVERALL: {samples[user_id][MFD_OVERALL]}")
        print(f"MFD_OVERALL_SD: {samples[user_id][MFD_OVERALL_SD]}")

        print(f"MSA_TRUE: {samples[user_id][MSA_TRUE]}")
        print(f"MSA_SD_TRUE: {samples[user_id][MSA_SD_TRUE]}")

        print(f"MSA_FALSE: {samples[user_id][MSA_FALSE]}")
        print(f"MSA_SD_FALSE: {samples[user_id][MSA_SD_FALSE]}")

        print(f"MSA_OVERALL: {samples[user_id][MSA_OVERALL]}")
        print(f"MSA_OVERALL_SD: {samples[user_id][MSA_OVERALL_SD]}")

        recognized_samples = len(samples[user_id][KNOWN])
        unrecognized_samples = len(samples[user_id][UNKNOWN])
        print(f"{recognized_samples} recognized images and {unrecognized_samples} unrecognized images")

        if args.show_plots:
            for index, sample in enumerate(samples[user_id][KNOWN]):
                print(f"Recognized sample {index + 1}/{recognized_samples}")
                plot_sample(sample)

            for index, sample in enumerate(samples[user_id][UNKNOWN]):
                print(f"Unrecognized sample {index + 1}/{unrecognized_samples}")
                plot_sample(sample)


if __name__ == "__main__":
    main()
