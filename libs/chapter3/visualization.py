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
Provides functions to visualize the obtained results.
'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import warnings

from libs.chapter3.model import Filter
from libs.chapter3.statistical_tests import pairwise_n_comparision, p_value_formatter

from plotly.subplots import make_subplots

pio.renderers.default = "notebook"
warnings.simplefilter(action='ignore', category=FutureWarning)


def _add_evolution_trace(fig, grouped_df, secondary=False, row=1, col=1, with_legend=True):
    means = grouped_df.mean(numeric_only=True)
    mins = grouped_df.min(numeric_only=True)
    maxs = grouped_df.max(numeric_only=True)
    stds = grouped_df.std(numeric_only=True)
    x = np.arange(1, len(means) + 1)
    
    if secondary: 
        line_color = 'purple'
        fill_color_range = 'rgba(106,13,173,0.4)'
        range_lines = dict(color='purple', dash='dash')
        fill_color_sd = 'rgba(255,255,0,0.4)'
        sd_lines = dict(color='yellow', dash='dot')
        legendgroup = 'secondary'
    else:
        line_color = 'red'
        fill_color_range = 'rgba(255,0,0,0.2)'
        range_lines = dict(color='red', dash='dash')
        fill_color_sd = 'rgba(0,0,255,0.2)'
        sd_lines = dict(color='purple', dash='dot')
        legendgroup = 'main'
        
    legendgrouptitle_text = f'<b>{means.index[0][1].upper()} {means.index[0][2].upper()}</b>'
    
    fig = fig.add_trace(
        go.Scatter(x=x, y=mins['value'], 
                   mode="lines", line=range_lines, fill='tonexty', fillcolor='rgba(0,0,0,0)', showlegend=False),
        secondary_y=secondary, row=row, col=col)
    fig = fig.add_trace(
        go.Scatter(x=x, y=maxs['value'], name="range", legendgroup=legendgroup, legendgrouptitle_text=legendgrouptitle_text, showlegend=with_legend,
                   mode="lines", line=range_lines, fill='tonexty', fillcolor=fill_color_range), 
        secondary_y=secondary, row=row, col=col)
    fig = fig.add_trace(
        go.Scatter(x=x, y=np.maximum((means - stds)['value'], np.zeros(22)), 
                   mode="lines", line=sd_lines, fill='tonexty', fillcolor='rgba(0,0,0,0)', showlegend=False),
        secondary_y=secondary, row=row, col=col)
    fig = fig.add_trace(
        go.Scatter(x=x, y=(means + stds)['value'], name="sd", legendgroup=legendgroup, showlegend=with_legend, 
                   mode="lines", line=sd_lines, fill='tonexty', fillcolor=fill_color_sd),
        secondary_y=secondary, row=row, col=col)
    fig = fig.add_trace(
        go.Scatter(x=x, y=means['value'], name="avg", legendgroup=legendgroup, showlegend=with_legend,
                   mode="lines", line_color=line_color), 
        secondary_y=secondary, row=row, col=col)
    return fig


def plot_evolution(reports, sources, filters, fig_titles, filters_secondary=None):
    '''
    Generates a figure containing a plot for each data source showing the accuracy/F1-score evolution with regards to the training data (i.e., n).
    
    Args:
        reports (`pandas.DataFrame`): Model reports.
        sources (`libs.chapter3.model.Source`[]): List with the data sources to include in the figure.
        filters (`libs.chapter3.model.Filter`): Filter to apply to the model reports.
        fig_titles (str[]): Title to use for the plot of each data source.
        filters_secondary (`libs.chapter3.model.Filter`): Filter to apply to the model reports, plotting the result in the secondary axis.  
        
    Returns:
        (plotly.Figure): Interactive Plotly figure.
    '''
    n_subplots = len(sources)
    fig = make_subplots(
        rows=n_subplots, 
        subplot_titles=[f'<b>{title}</b>' for title in fig_titles], 
        x_title='<b>Number of subjects in training</b>',
        y_title=f"<b>{str(filters.target).capitalize()} {filters.metric}</b>",
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}]] * n_subplots)
    #fig = go.Figure()

    df_filtered = filters.apply(reports)
    if filters_secondary:
        df_filtered_secondary = filters_secondary.apply(reports)
        max_values = []
        
    for i, source in enumerate(sources):
        row = i+1
        with_legend = row == 1
        df_source = Filter(None, source, None, None).apply(df_filtered)
        #df_source = df_filtered[df_filtered.source == str(source)]
        grouped_df = df_source.groupby(['n', 'target', 'metric'])
        fig = _add_evolution_trace(fig, grouped_df, row=row, with_legend=with_legend)

        if filters_secondary:
            df_source = Filter(None, source, None, None).apply(df_filtered_secondary)
            grouped_df = df_source.groupby(['n', 'target', 'metric'])
            max_values.append(grouped_df.max(numeric_only=True).loc[22,'model','training time']['value'])
            fig = _add_evolution_trace(fig, grouped_df, True, row=row, with_legend=with_legend)

    for ax in fig['layout']:
        if ax[:5]=='xaxis':
            fig['layout'][ax]['dtick'] = 1
        if ax[:5]=='yaxis':
            if fig['layout'][ax]['side'] != 'right':
                #fig['layout'][ax]['dtick'] = 0.2
                fig['layout'][ax]['range'] = [-0.01,1]
            elif filters_secondary:
                fig['layout'][ax]['range'] = [-0.01, max(max_values)]
            
    fig.update_layout(
        height=800, #width=1000,
        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.95
        )
    )
    
    if filters_secondary:
        fig.add_annotation(
            text=f"<b>{str(filters_secondary.target).capitalize()} {filters_secondary.metric} (s)</b>",
            font_size=16,
            textangle= -90,
            x= 0,
            xanchor= 'left',
            xref= 'paper',
            xshift= 820,
            y= 0.5,
            yanchor= 'middle',
            yref= 'paper',
            showarrow=False
        )
            
    return fig


def plot_comparison(reports, models, sources, filters, sources_print):
    '''
    Generates a figure containing a plot for each data source and model showing the accuracy/F1-score evolution with regards to the training data (i.e., n).
    
    Args:
        reports (pandas.DataFrame): Model reports.
        models (`libs.chapter3.model.Models`[]): List with the models to include in the figure.
        sources (`libs.chapter3.model.Source`[]): List with the data sources to include in the figure.
        filters (`libs.chapter3.model.Filter`): Filter to apply to the model reports.
        sources_print (dict): Mapping between a Source and a string representation.

    Returns:
        (`plotly.Figure`): Interactive Plotly figure.
    '''
    
    fig = make_subplots(cols=4, rows=3, 
                        x_title='<b>Number of subjects in training</b>', 
                        y_title=f'<b>{str(filters.metric).capitalize()}</b>', 
                        subplot_titles=[ f'<b>{str(model_type).upper()}</b>' for model_type in models],
                        shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=0.08)
    
    df = filters.apply(reports)
    for i, model_type in enumerate(models):
        for j, data_source in enumerate(sources):
            col = i+1
            row = j+1
            with_legend = row == 1 and col == 1
            
            df_fil = Filter(model_type, data_source, None,  None).apply(df)
            grouped_df = df_fil.groupby(['n', 'target', 'metric'])
            fig = _add_evolution_trace(fig, grouped_df, row=row, col=col, with_legend=with_legend)
            fig.update_yaxes(title=f'<b>{sources_print[data_source]}</b>' if col == 1 else '', dtick = 0.2, range=[-0.01, 1], row=row, col=col)
            fig.update_xaxes(dtick = 1, col=col)
    
    fig.update_layout(
        height=700,
        #width=1500,
        #height=800,
        #width=1000,
        yaxis_range=[-0.01,1],
        showlegend=True,
        legend = dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )
    
    fig['layout']['annotations'][-1]['xshift'] = -60
    
    return fig




def plot_pairwise_comparision(reports, sources, filters, sources_print, alternative='two-sided', stars=False, parametric=False):
    '''
    Generates a figure to represent the pairwise tests generated by `libs.chapter3.statistical_tests.pairwise_n_comparison`
    
    Args:
        reports (`pandas.DataFrame`): Model reports.
        sources (`libs.chapter3.model.Source`[]): List with the data sources to include in the figure.
        filters (`libs.chapter3.model.Filter`): Filter to apply to the model reports.
        sources_print (dict): Mapping between a Source and a string representation.
        alternative (str): Hypothesis to test. One of: 'two-sided', 'less' or 'greater'.
        stars (boolean): Replace p-values under 0.05 by stars. '*' when 0.01<p-value<0.05; '**' when 0.001<p-value<0.01; '***' when p-value<0.001;  
        parametric (boolean): Compute parametric or non-parametric tests.

    Returns:
        (`plotly.Figure`): Interactive Plotly figure.
    '''


    def mean_to_categorical(x):
        if isinstance(x, str):
            return x
        return int(x < 0)   

    figure_index = np.arange(1,23)
    fig = make_subplots(rows=len(sources), cols=1, 
                        x_title='<b>Number of subjects (<i>n<sub>2</sub></i>)</b>', y_title='<b>Number of subjects (<i>n<sub>1</sub></i>)</b>',
                        subplot_titles=[f'<b>{sources_print[source]}</b>' for source in sources],
                        vertical_spacing=0.05)
    
    axes_config = {'dtick': 1, 'gridcolor': 'darkgrey', 'gridwidth': 2 }
    
    model_data = filters.apply(reports)
    for i, source in enumerate(sources):
        source_data = Filter(None, source, None, None).apply(model_data)
        
        grouped_data = source_data.groupby("n")
        means = grouped_data.mean(numeric_only=True)["value"].values if parametric else grouped_data.median(numeric_only=True)["value"].values
        pairwise_means = means - means[:, None]
        np.fill_diagonal(pairwise_means, 101)
        pairwise_means = pd.DataFrame(np.triu(pairwise_means), index=figure_index, columns=figure_index)
        pairwise_means = pairwise_means.replace(101, "").applymap(mean_to_categorical).replace('', np.nan)
        
        pairwise_matrix = pairwise_n_comparision(reports, Filter(filters.model, source, filters.target, filters.metric), alternative=alternative, stars=False, parametric=parametric)
        pairwise_matrix = pairwise_matrix.where(np.triu(np.ones_like(pairwise_matrix, dtype=bool))).replace('-', np.nan).astype(float)
    
        color_value = pairwise_matrix * 0.4 + pairwise_means * 0.6 
        pairwise_matrix = pairwise_matrix.fillna('').applymap(p_value_formatter(True)) if stars else pairwise_matrix
        
        fig.add_trace(
            go.Heatmap(
                z=color_value,
                x=figure_index,
                y=figure_index,
                hoverongaps = False,
                hovertemplate='B: %{x}<br>A: %{y}<br>p-value: %{text}<extra></extra>',
                text=pairwise_matrix,
                texttemplate="%{text}", 
                textfont_size=13,
                xgap = 2, ygap = 2,
                #colorscale=[[0, 'rgb(153,0,0)'], [0.4, 'rgb(255,204,204)'], [0.6, 'rgb(0,153,0)'], [1, 'rgb(204,255,204)']],
                colorscale=[[0, 'rgb(33,113,181)'], [0.4, 'rgb(222,235,247)'], [0.6, 'rgb(178,24,43)'], [1, 'rgb(253,219,199)']],
                #rgb(222,235,247), 'rgb(33,113,181)'
                showscale=False,
            ), row=i+1, col=1,
        )
        
        fig.update_xaxes(axes_config, row = i+1, col = 1)
        fig.update_yaxes(axes_config, row = i+1, col = 1)
    
    fig.update_layout(
        height = 1200, #width = 1200, 
        plot_bgcolor = 'rgba(0,0,0,0)',
    )
    return fig