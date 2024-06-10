# Copyright 2024 Miguel Matey Sanz
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Utils to compute the Bland-Altman components.
'''

import numpy as np

from scipy import stats

def bland_altman_components(data1, data2):
    '''
    Computes components of the Bland-Altman analysis. 
    
    Args:
        data1 (list[float]): Measures from method 1.
        data2 (list[float]): Measures from method 2.

    Returns:
        mean (list[float]): mean between the ith sample of both methods.
        difference (list[float]): difference between the ith sample of both methods.
        mean_difference (float): mean difference of all measurements.
        sd (float): standard deviation of the differences.
        high (float): Higher limit of agreement (mean_difference + 1.96*sd)
        low (float): Lower limit of agreement (mean_difference - 1.96*sd)
        ci (dict): Dict containing the 95% confidence interval for the mean difference and the limits of agreement.
    '''

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