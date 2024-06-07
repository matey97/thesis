import numpy as np

from scipy import stats

def bland_altman_components(data1, data2):
    n = data1.size
    dof = n - 1
    mean = np.mean([data1, data2], axis=0)
    difference = data1 - data2                   
    mean_difference = np.mean(difference)                   
    sd = np.std(difference, axis=0)
    mean_difference_se = np.sqrt(sd**2 / n)
    
    high = mean_difference + 1.96 * sd
    low = mean_difference - 1.96 * sd
    high_low_se = np.sqrt(3 * sd**2 / n)
    
    ci = dict()
    ci["mean"] = stats.t.interval(0.95, dof, loc=mean_difference, scale=mean_difference_se)
    ci["high"] = stats.t.interval(0.95, dof, loc=high, scale=high_low_se)
    ci["low"] = stats.t.interval(0.95, dof, loc=low, scale=high_low_se)

    return (mean, difference, mean_difference, sd, high, low, ci)