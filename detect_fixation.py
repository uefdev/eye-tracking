#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
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


def zip_coords(coordinate_vector):
    return list(zip(coordinate_vector[0::2], coordinate_vector[1::2]))


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

    return reduce(
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
        )
        ),
        dict({user_id: dict({KNOWN: [], UNKNOWN: []}) for user_id in target_users})
    )


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
        "--filename",
        help="Filename inside the ZIP archive",
        type=str,
        dest="file_name",
        default="train.csv"
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
    args = parser.parse_args()

    samples = get_filtered_data_as_dict(
        archive_name=args.archive_name,
        file_name=args.file_name,
        target_users=args.target_users,
        dispersion_threshold=args.dispersion_threshold,
        duration_threshold=args.duration_threshold
    )

    for user_id in samples:
        print(f"\nUser {user_id}")

        recognized_samples = len(samples[user_id][KNOWN])
        unrecognized_samples = len(samples[user_id][UNKNOWN])
        print(f"{recognized_samples} recognized images and {unrecognized_samples} unrecognized images")

        for index, sample in enumerate(samples[user_id][KNOWN]):
            print(f"Recognized sample {index + 1}/{recognized_samples}")
            plot_sample(sample)

        for index, sample in enumerate(samples[user_id][UNKNOWN]):
            print(f"Unrecognized sample {index + 1}/{unrecognized_samples}")
            plot_sample(sample)


if __name__ == "__main__":
    main()
