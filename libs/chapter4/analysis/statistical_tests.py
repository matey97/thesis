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

pg.options['round.column.p-val'] = 3

# Computed MWU power using G*Power 3.1:
# - Test family > t tests
# - Statistical test > Means: Wilcoxon-Mann_Whitney test (two groups)
# - Type of power analysis > Post hoc: Compute achieved power -- given alpha, sample size, and effect size.
COMPUTED_POWERS = {
    'sp_accuracy': 1
}

def compare_splitting_approaches(reports, metrics):
    '''
    Statistically compares the two splitting approaches on smartphone and smartwatch data. More concretely,
    determines if there is a significant difference in the accuracy of the models or the F1-score of the
    `TURNING`, `SITTING_DOWN` and `TURN_TO_SIT` activities.
    
    Args:
        reports (`dict`): Model reports.
        metrics (list[str]): Performance metric to compare.
        
    Returns:
        (`pandas.DataFrame`): DataFrame containing the statistical test results.
    '''

    def test_builder(parametric, source, metric, alternative='two-sided'):
        def mwu(g1, g2):
            res = pg.mwu(g1, g2, alternative=alternative).loc['MWU']
            power = COMPUTED_POWERS[f'{source}_{metric}']
            return [np.round(np.median(g1), 3), np.round(np.median(g2), 3), f'U={res["U-val"]}, p-val={res["p-val"]}, power={power}']

        def ttest(g1, g2):
            res = pg.ttest(g1, g2, alternative=alternative).loc['T-test']
            return [np.round(np.mean(g1), 3), np.round(np.mean(g2), 3), f't({res["dof"]})={res["T"]}, p-val={res["p-val"]}, power={res["power"]}']

        return ttest if parametric else mwu
    
    results = []
    for source, datasource_report in reports.items():
        for metric in metrics:
            partial_result = [source, metric]
            if metric not in datasource_report['ts']:
                results.append(partial_result + ['-', np.round(np.mean(datasource_report['tts'][metric]), 3), '-'])
            elif metric not in datasource_report['tts']:
                results.append(partial_result + [np.round(np.mean(datasource_report['ts'][metric]), 3), '-', '-'])
            else:
                ts_dataset = datasource_report['ts'][metric]
                tts_dataset = datasource_report['tts'][metric]

                tester = test_builder(bool(pg.normality(ts_dataset)['normal'].values[0]) and bool(pg.normality(tts_dataset)["normal"].values[0]), source, metric)
                results.append(partial_result + tester(ts_dataset, tts_dataset))
    return pd.DataFrame(results, columns=['source', 'metric', 'turning_sitting', 'turn_to_sit', 'two-tailed test'])