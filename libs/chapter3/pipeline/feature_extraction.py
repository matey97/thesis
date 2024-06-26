import numpy as np

from functools import reduce


def apply_feature_extraction(window):
    means = np.apply_along_axis(np.mean, 1, window)
    medians = np.apply_along_axis(np.median, 1, window)
    maxs = np.apply_along_axis(np.max, 1, window)
    mins = np.apply_along_axis(np.min, 1, window)
    stds = np.apply_along_axis(np.std, 1, window)
    ranges = maxs - mins
    rmss = np.apply_along_axis(rms, 1, window)
    pitch_rolls = pitch_and_roll(window)
    angles = compute_gyro_angle(window)

    return np.concatenate((means, medians, maxs, mins, stds, ranges, rmss, pitch_rolls, angles))

def rms(values):
    return np.sqrt(reduce(lambda prev, curr: prev + curr ** 2, values, 0) / len(values))

    
def pitch_and_roll(window):
    def angular_function(a, b):
        return np.arctan2(a, b) * 180/np.pi

    return np.array([
        np.mean(angular_function(window[1], window[2])),
        np.mean(angular_function(-window[0], np.sqrt(np.power(window[1], 2) + np.power(window[2], 2))))
    ])

    
def integrator(series):
    return np.trapz(series)


def compute_gyro_angle(window):
    return np.array([
        integrator(window[3]), integrator(window[4]), integrator(window[5])
    ])