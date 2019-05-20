# -*- coding: utf-8 -*-
"""eye-tracking-project
 
Automatically generated by Colaboratory.
 
Original file is located at
   https://colab.research.google.com/drive/1uOSbXJ3q_LPRi8Baz7_75p3yJR_CaerT
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
# Group 8 - s8, s18, s28, s4, s14, s24
 
"""
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
### TODO csv data source file location & reading ### 
from google.colab import files
uploaded = files.upload()
 
data = uploaded['filtered.csv']
data = data.decode("utf-8")
 
print(type(data))
 
csv_data = list(map(
    lambda row: row.split(","),
    data.split("\n")
))
 
fixed_data = list(filter(
    lambda x: len(x) > 1,  
    map(
        lambda row: [
            str(value) if index == 0 or index == 1 else float(value) for index, value in enumerate(row)
        ],
    csv_data
)))
 
 
def zip_coords(coordinate_vector):
    return list(zip(coordinate_vector[0::2], coordinate_vector[1::2]))
 
 
coord_data = list(map(
    lambda row: [row[0], row[1], zip_coords(row[2:])],
    fixed_data
))
 
print(len(coord_data))
print(coord_data[0])
 
a = np.mean(coord_data[0][2], axis=0)
print(a)
 
def calculate_dispersion(points):
    x_max = max(points, key=lambda pair: pair[0])[0]
    y_max = max(points, key=lambda pair: pair[1])[1]
    x_min = min(points, key=lambda pair: pair[0])[0]
    y_min = min(points, key=lambda pair: pair[1])[1]
    return (x_max - x_min) + (y_max - y_min)
 
 
# Identification by dispersion threshold
def detect_fixation(original_points, dispersion_threshold = 80, duration_threshold = 100):
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
                #return fixation_points
 
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
 
sample = coord_data[50]
 
fixation_points = detect_fixation(sample[2], 80, 100)
saccade_amplitudes = calculate_saccade_amplitudes(sample[2], fixation_points)
 
plot_fixation = np.concatenate(np.array(fixation_points)[:,1]).reshape(-1, 2)
 
sample_array = np.array(sample[2])
plt.scatter(sample_array[:,0], sample_array[:,1])
plt.scatter(plot_fixation[:,0], plot_fixation[:,1])
plt.show()