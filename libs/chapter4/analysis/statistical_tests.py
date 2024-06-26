'''
Provides functions to compute statistical tests to determine the significance of the obtained results.
'''


import numpy as np
import pandas as pd
import pingouin as pg

from libs.chapter4.analysis.tug_results_processing import compute_rmse_by_subject, DURATION_AND_PHASES

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


def compare_distribution_with_zero(distribution):
    '''
    Statistically compares the mean of a distribution with 0 using a T-test (normal data) or a W-test (non-normal data).

    Args:
        distribution (list[float]): array of numbers constituting the distribution.

    Returns:
        p_value (float): P-value of the test. A value less than 0.05 indicates a significant difference between the distribution mean and zero.
        is_normal (bool): Indicates if the distribution is normal or not and, therefore, which statistical test was used.
    '''

    is_normal = pg.normality(distribution).loc[0, 'normal']
    return pg.ttest(distribution, 0).loc['T-test']['p-val'] if is_normal else pg.wilcoxon(distribution).loc['Wilcoxon']['p-val'], is_normal


# Computed MWU power using G*Power 3.1:
# - Test family > t tests
# - Statistical test > Means: Wilcoxon-Mann_Whitney test (two groups)
# - Type of power analysis > Post hoc: Compute achieved power -- given alpha, sample size, and effect size.
COMPUTED_POWERS = {
    'duration': 0.542,
    'first_walk': 0.338,
    'first_turn': 0.434,
    'second_walk': 0.283,
    'second_turn': 0.197,
    'sitting_down': 0.603
}


def compare_rmse_distributions(errors_df):
    '''
    Statistically compares the RMSE of two distributions using a two-sample T-test (normal data) or a MWU test (non-normal data).
    The compared distributions are the inter-subject RMSE of the TUG duration and each subphase.

    Args:
        errors_df (`pandas.DataFrame`): DataFrame containing the error in ms of the system measures and the reference method for all subjects.

    Returns:
        (`pandas.DataFrame`): DataFrame containing the tests results of comparing each measure (i.e., TUG duration and subphases).
    '''

    rmse_df = compute_rmse_by_subject(errors_df)
    df_reset = rmse_df.reset_index()
    results = []
    for measure in DURATION_AND_PHASES:
        c1_df = df_reset[df_reset.system == 'C1'][measure].values
        c2_df = df_reset[df_reset.system == 'C2'][measure].values
        result = [measure]
        if pg.normality(c1_df).loc[0, 'normal'] and pg.normality(c2_df).loc[0, 'normal']:
            ttest = pg.ttest(c1_df, c2_df).loc['T-test']
            ttest = f't({ttest["dof"]:.2f})={ttest["T"]:.2f}, p-val={ttest["p-val"]}, power={ttest["power"]}'
            result += [np.mean(c1_df), np.mean(c2_df), ttest]
        else:
            mwu = pg.mwu(c1_df, c2_df).loc['MWU']
            power = COMPUTED_POWERS[measure]
            mwu = f'U={mwu["U-val"]}, p-val={mwu["p-val"]}, power={power}'
            result += [np.median(c1_df), np.median(c2_df), mwu]

        results.append(result)
    return pd.DataFrame(results, columns=['Measure', 'M(C1)', 'M(C2)', 'Test'])



def compute_icc(system_results, manual_results, labels, icc_type='ICC2'):
    '''
    Computes the Intraclass Correlation Coefficient (ICC) between the TUG results obtained by the system and
    the reference methods for each TUG measure.
    
    Args:
        system_results (list[`pandas.DataFrame`]): Contains the DataFrames with the measures generated by both system's configurations.
        manual_results (`pandas.DataFrame`): DataFrame containing the measures generated by the reference method.
        labels (list[str]): Text labels associated with the DataFrame in `system_results`. 
        icc_type (str): Type of ICC to compute. One of: ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k.

    Returns:
        (pd.DataFrame): DataFrame containing the ICC results for each TUG measure and system (`C1` and `C2`).
    '''

    def join_and_melt(df1, df2):
        df = pd.concat([df1[df1.status == 'success'], df2[df1.status == 'success']], ignore_index=True)
        df = df.drop(['status'], axis=1)
        return df.melt(id_vars=["subject", "system"], var_name='phase', value_name='ms')
    
    dfs = []
    manual_results = manual_results.copy()
    manual_results['system'] = 'manual'
    for system_result, label in zip(system_results, labels):
        system_result = system_result.copy()
        system_result['system'] = label
        df = join_and_melt(system_result, manual_results)
        dfs.append(df)
        
    icc_results = []
    for phase in DURATION_AND_PHASES:
        for df in dfs:
            icc = pg.intraclass_corr(df[df.phase == phase], targets='subject', raters='system', ratings='ms', nan_policy='omit').round(3)
            icc = icc[icc['Type'] == icc_type].iloc[0] 
                
            icc_value = icc['ICC']
            ci = icc['CI95%']
            f_test = icc['F']
            p_value = icc['pval']
            
            
            icc_results.append([df.loc[0].system, phase, icc_value, ci, f_test, p_value])
    
    icc_df = pd.DataFrame(icc_results, columns=['system', 'phase', 'ICC', 'CI', 'F Test', 'p-value']).set_index(['phase', 'system'])
    return icc_df