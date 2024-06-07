import math

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from libs.chapter4.analysis.bland_altman import bland_altman_components
from libs.chapter4.analysis.tug_results_processing import DURATION_AND_PHASES
from libs.chapter4.analysis.statistical_tests import compare_distribution_with_zero

from plotly.subplots import make_subplots

pio.renderers.default = "notebook"


def _create_figure_with_subplots(x_title, y_title, subplots, with_titles, max_subfigures_in_row=3):
    def compute_subplots_rows_cols(n_subplots, max_subfigures_in_row):
        cols = n_subplots if n_subplots <= max_subfigures_in_row else max_subfigures_in_row
        rows = math.ceil(n_subplots / cols)
        return rows, cols
    
    rows, cols = compute_subplots_rows_cols(len(subplots), max_subfigures_in_row)
    return make_subplots(
        rows=rows, cols=cols, 
        x_title=x_title, y_title=y_title, subplot_titles=subplots if with_titles else None,
        shared_xaxes=False, shared_yaxes=True, vertical_spacing=0.1
    ), rows, cols


def _compute_assigned_row_col(index, cols):
    row = index // cols + 1
    col = index % cols + 1
    return row, col


def error_distribution(errors_df):
    def pval_formatter(val):
        if val < 0.001:
            return '<.001'
        return f'{val:.3f}'
    
    errors_df_tidy = errors_df[errors_df.status == 'success'].melt(id_vars=["subject", "system", "status"], var_name='phase', value_name='error')
    fig = px.box(errors_df_tidy, x='phase', y='error', color='system', height=400, #width=1000, 
                 labels={"phase": "Measure", "error": "Error (ms)", "system": "Config."})
    
    fig.update_xaxes(title_font_size=20)
    fig.update_yaxes(title_font_size=20)

    c1_errors = errors_df[(errors_df.status == 'success') & (errors_df.system == 'C1')]
    c2_errors = errors_df[(errors_df.status == 'success') & (errors_df.system == 'C2')]
    for phase in DURATION_AND_PHASES: 
        c1_err = c1_errors[phase].to_numpy()
        c2_err = c2_errors[phase].to_numpy()
        
        c1_test, c1_normal = compare_distribution_with_zero(c1_err / 1000)
        c2_test, c2_normal = compare_distribution_with_zero(c2_err / 1000)
        
        c1_pval = pval_formatter(c1_test)
        c2_pval = pval_formatter(c2_test)
        fig.add_annotation(x=phase, y=-3000, text=f'<b>{c1_pval}</b>', showarrow=False, xshift=-25, bgcolor='lightgreen' if c1_normal else 'white', bordercolor='black', font_size=14)
        fig.add_annotation(x=phase, y=-3000, text=f'<b>{c2_pval}</b>', showarrow=False, xshift=25, bgcolor='lightgreen' if c2_normal else 'white', bordercolor='black', font_size=14)
    
    fig.update_layout(
        plot_bgcolor = 'rgb(255,255,255)',
        yaxis_gridcolor = 'darkgrey',
        yaxis_gridwidth = 1,
        yaxis_dtick = 1000,
        yaxis_zeroline=True,
        yaxis_zerolinecolor='black',
        yaxis_zerolinewidth = 2, 
        boxgroupgap=0.1,
        boxgap=0.1
    )

    return fig


def bland_altman_plot(system_results, man_results, system_desc, attrs, with_titles=True, limit_y_axis_to=None):  
    x_title=f'Mean {system_desc} and manual measures (s)'
    y_title=f'Difference {system_desc} and manual measures (s)'
    fig, rows, cols = _create_figure_with_subplots(x_title, y_title, attrs, with_titles)
    fig.update_annotations(font_size=20)
    
    df = system_results.join(man_results, lsuffix='_system', rsuffix='_manual')
        
    for i, attr in enumerate(attrs):             
        row, col = _compute_assigned_row_col(i, cols)
        system_attr = f'{attr}_system'
        manual_attr = f'{attr}_manual'
        attr_df = df[df.status == 'success']
        
        mean, difference, mean_difference, sd, high, low, ci = bland_altman_components(attr_df[system_attr] / 1000, attr_df[manual_attr] / 1000)    
        
        scatter = go.Scatter(x=mean, y=difference, mode='markers')
        fig.append_trace(scatter, row=row, col=col)
        fig.add_hline(y=mean_difference, line_color='red', line_width=2, annotation={
            'text': '<b>Mean: {0:.3f}</b>'.format(mean_difference),
            'font_size': 20
        }, row=row, col=col)
        fig.add_hline(high, line_dash='dash', line_color='red',  annotation={
            'text': '<b>+1.96SD: {0:.3f}</b>'.format(mean_difference + 1.96*sd),
            'font_size': 20
        }, row=row, col=col)
        fig.add_hline(low, line_dash='dash', line_color='red', annotation={
            'text': '<b>-1.96SD: {0:.3f}</b>'.format(mean_difference -1.96*sd),
            'font_size': 20
        }, annotation_position='bottom right', row=row, col=col)
        fig.add_hrect(ci['mean'][0], ci['mean'][1], line_width=0, fillcolor="red", opacity=0.2, row=row, col=col)
        fig.add_hrect(ci['high'][0], ci['high'][1], line_width=0, fillcolor="red", opacity=0.2, row=row, col=col)
        fig.add_hrect(ci['low'][0], ci['low'][1], line_width=0, fillcolor="red", opacity=0.2, row=row, col=col)
    
    for ax in fig['layout']:
        if ax[:5] == 'yaxis':
            fig['layout'][ax]['dtick'] = 0.5
            fig['layout'][ax]['scaleanchor'] = 'y'
            fig['layout'][ax]['gridcolor'] = 'darkgrey'
            fig['layout'][ax]['gridwidth'] = 1
            fig['layout'][ax]['zeroline'] = True
            fig['layout'][ax]['zerolinecolor'] = 'black'
            fig['layout'][ax]['zerolinewidth'] = 2
            fig['layout'][ax]['range'] = limit_y_axis_to
        elif ax[:5] == 'xaxis':
            fig['layout'][ax]['gridcolor'] = 'darkgrey'
            fig['layout'][ax]['gridwidth'] = 1
    
    fig.update_layout(
        title={
            'text': 'Bland-Altman Plot',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 22
        },
        #width=400 * cols if len(attrs) > 1 else 800,
        height=400 * rows if len(attrs) > 1 else 600,
        plot_bgcolor = 'rgb(255,255,255)',
        yaxis_gridcolor = 'darkgrey',
        yaxis_gridwidth = 1,
        yaxis_dtick = 0.5,
        xaxis_gridcolor = 'darkgrey',
        xaxis_gridwidth = 1,
        yaxis_zeroline=True,
        yaxis_zerolinecolor='black',
        yaxis_zerolinewidth = 2, 
        showlegend=False,
    )
    
    return fig
