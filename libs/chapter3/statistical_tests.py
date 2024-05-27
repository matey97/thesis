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

from libs.chapter3.model import ActivityMetric, Filter, ModelMetric, TargetFilter, Source 

pg.options['round.column.p-val'] = 3
pg.options['round.column.p-unc'] = 3


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
        models (list[`libs.chapter3.model.Models`]): List with the models.
        sources (list[`libs.chapter3.model.Source`]): List with the data sources.
        
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


def statistical_comparison(reports, metric_filter, focus_on, groups, alternative='two-sided'):
    '''
    Args:
        reports (`pandas.DataFrame`): Model reports.
        metric_filter (tuple[str, str]): Metric filter to apply to the model reports.
        focus_on (list[str]): Items being compared. List items are one of `libs.chapter3.model.Source` or `libs.chapter3.model.Model`.
        groups (list[str]): Each group where `focus_on` items are compared. List items are one of `libs.chapter3.model.Source` or `libs.chapter3.model.Model`.
        alternative (str): Hypothesis to test. One of: 'two-sided', 'less' or 'greater'.
        
    Returns:
        (`pandas.DataFrame`): DataFrame containing groups comparisons.
    '''

    def filter_builder(target, metric):
        def filter_by_model(focus):
            return Filter(focus, None, target, metric)
        def filter_by_source(focus):
            return Filter(None, focus, target, metric)
            
        return filter_by_model if isinstance(groups[0], Source) else filter_by_source
    
    test_results = []
    posthoc_results = []  
    target, metric = metric_filter
    filterer = filter_builder(target, metric)
    for focus in focus_on:
        comparision, posthoc = _compare_groups(reports, filterer(focus), groups, alternative)
        comparision = comparision.set_index('n')
        #comparision['activity'] = target
        posthoc['focus'] = focus
        test_results.append(comparision)
        posthoc_results.append(posthoc)
    #return posthoc_results    
    df_test = pd.concat(test_results, axis=1, keys=focus_on)

    #columns = df_test.columns.to_list()
    #columns.remove('n')
    #columns.remove('activity')
    #df_test = df_test.sort_values(['n', 'activity']).reset_index(drop=True)[['n', 'activity'] + columns]
    #df_test = df_test.pivot(index='n', columns='activity', values=columns).stack(0).unstack()
    #df_test = df_test.reindex(columns=df_test.columns.reindex(columns, level=1)[0])
    
    df_posthoc = pd.concat(posthoc_results, axis=0)
    df_posthoc = df_posthoc.set_index(['focus', 'n'])
    
    return df_test, df_posthoc


def _compare_groups(reports, filters, groups, alternative='two-sided'):
    filtered_reports = filters.apply(reports)
    
    between = 'source' if isinstance(groups[0], Source) else 'model'
    results = []
    posthoc_results = []
    for n in range(1, 23):
        n_reports = filtered_reports[filtered_reports.n == n]
        kruskal = pg.kruskal(n_reports, dv='value', between=between, detailed=True).loc['Kruskal']
        medians = n_reports.groupby(between)['value'].median().round(3)
        
        test_results = []
        labels = ['n']
        for group in groups:
            test_results += [medians[str(group)]]
            labels += [str(group)]
        test_results += [kruskal["H"].round(3), kruskal['p-unc']]
        labels += [f'H({kruskal["ddof1"]})', 'p-value']
        results.append([n] + test_results)
        
        if kruskal['p-unc'] <= 0.05:
            posthoc = pg.pairwise_tests(n_reports, dv='value', between=between, parametric=False, return_desc=True)[['A', 'B', 'mean(A)', 'std(A)', 'mean(B)', 'std(B)', 'U-val', 'p-unc']]
            posthoc['n'] = n
            posthoc_results.append(posthoc)
 
    posthoc_df = pd.concat(posthoc_results)#.set_index(['n'])
    posthoc_df = posthoc_df.rename(columns={'U-val': 'U', 'p-unc': 'p-value'})
    return pd.DataFrame(results, columns=labels), posthoc_df