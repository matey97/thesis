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
Provides functions to compute statistical tests to determine the significance of the obtained results.
'''

import numpy as np
import pandas as pd
import pingouin as pg

from libs.chapter3.model import ActivityMetric, Filter, ModelMetric, TargetFilter 


def pairwise_n_comparision(data, filters, alternative='two-sided', stars=False, parametric=False):
    '''
    Computes pairwise tests for each value of n.
    
    Args:
        data (`pandas.DataFrame`): Model reports.
        filters (str): Filter to apply to the model reports. See: `libs.chapter3.model.Filter`
        alternative (str): Hypothesis to test. One of: 'two-sided', 'less' or 'greater'.
        stars (boolean): Replace p-values under 0.05 by stars. '*' when 0.01<p-value<0.05; '**' when 0.001<p-value<0.01; '***' when p-value<0.001;  
        parametric (boolean): Compute parametric or non-parametric tests.
        
    Returns:
        (`pandas.DataFrame`): DataFrame containing the pairwise tests.
    '''
    data_filtered = filters.apply(data)
    
    if parametric:
        df = data_filtered[['n', 'i', 'value']].pivot(index='i', columns='n')['value']
        return df.ptests(stars=stars, alternative=alternative)
    
    non_parametric_pariwise = data_filtered.pairwise_tests(dv='value', between='n', parametric=False, alternative=alternative)

    mat = np.empty((22, 22))
    mat = mat.astype(str)

    formatter = p_value_formatter(stars)
    for i in range(1, 22):
        for j in range(i + 1, 23):
            row = non_parametric_pariwise[(non_parametric_pariwise.A == i) & (non_parametric_pariwise.B == j)]
            mat[i - 1,j - 1] = formatter(row['p-unc'].to_numpy()[0])
            mat[j - 1,i - 1] = row['U-val'].to_numpy()[0]
    np.fill_diagonal(mat, "-")
    index = np.arange(1, 23)
    return pd.DataFrame(mat, index=index, columns=index)


def p_value_formatter(stars, decimals=3):
    def as_value(val):
        return f'{val:.{decimals}}'
    def as_stars(val):
        if isinstance(val, str):
            try:
                val = float(val)
            except:
                return val
            
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return f'{val:.{decimals}}'
        
    return as_stars if stars else as_value


def is_parametric_data(reports, models, sources):
    '''
    Determines if the results in the reports follow a parametric or a non-parametric distribution.

    Args:
        reports (`pandas.DataFrame`): Model reports.
        models (`libs.chapter3.model.Models`[]): List with the models.
        sources (`libs.chapter3.model.Source`[]): List with the data sources.
        
    Returns:
        (`pandas.DataFrame`): DataFrame indicating if the results from a model+source are parametric (`True`) or not (`False`).
    '''

    dfs = []
    for model in models:
        for source in sources:
            df = _check_normality(Filter(model, source, None, None).apply(reports)).all().to_frame().transpose()
            df['model'] = model
            df['source'] = source
            dfs.append(df)

    return pd.concat(dfs).set_index(['model', 'source'])


def _check_normality(data):
    to_check = [(TargetFilter.MODEL, ModelMetric.ACCURACY), 
            (TargetFilter.SEATED, ActivityMetric.F1),
            (TargetFilter.STANDING_UP, ActivityMetric.F1),
            (TargetFilter.WALKING, ActivityMetric.F1),
            (TargetFilter.TURNING, ActivityMetric.F1),
            (TargetFilter.SITTING_DOWN, ActivityMetric.F1)]

    normality_table = np.full((22, 6), False)
    columns = []

    for i, (target_filter, metric) in enumerate(to_check):
        filtered = Filter(None, None, target_filter, metric).apply(data)
        columns.append(f'{target_filter} {metric}')
        for n in range(1, 23):
            filtered_n = filtered[filtered.n == n]
            normal = pg.normality(filtered_n['value']).loc['value']['normal']
            if normal:
                normality_table[n-1, i] = True

    return pd.DataFrame(normality_table, index=np.arange(1,23), columns=columns)