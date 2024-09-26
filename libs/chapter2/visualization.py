'''
Provides a function to plot the collected data.
'''


import plotly.graph_objects as go

from plotly.subplots import make_subplots

from libs.common.plotly import configure_renderers
configure_renderers()


def plot_execution(data_collection, execution):
    '''
    Generates an interactive plot with the accelerometer and gyroscope data of the specified execution.
    
    Args:
        data_collection (dict): Dict containing the collected dataset. See: `utils.data_loading.load_data()`
        execution (str): execution data to plot. Format: 'sXX_YY_{sp|sw}'
        
    Returns:
        figure (`plotly.graph_objs.Figure`): Interactive plot
    '''
    
    if execution not in data_collection:
        raise Exception(f'Execution {execution} not present in dataset.')
        
    df = data_collection[execution]
    components_group = [['x_acc', 'y_acc', 'z_acc'], ['x_gyro', 'y_gyro', 'z_gyro']]
    y_axes_titles = ['Intensity (m/s<sup>2</sup>)', 'Angular velocity (º/s)']
    figures = []
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    line_style = ['solid', 'dashdot', 'dot']
    line_colors = ['red', 'green', 'blue']
    
    change = df['label'].shift(1, fill_value=df["label"].head(1)) != df["label"]
    change_timestamps = df[change]['timestamp'].to_list()
    change_timestamps = df.head(1)['timestamp'].to_list() + change_timestamps + df.tail(1)['timestamp'].to_list()
    activities = ['SEATED', 'STANDING UP', 'WALKING', 'TURNING', 'WALKING', 'TURNING', 'SITTING DOWN', 'SEATED']
    
    for c, components in enumerate(components_group):
        
        for i, component in enumerate(components):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df[component], 
                    line=go.scatter.Line(dash=line_style[i], color=line_colors[i]),
                    name=component.split('_')[0], legendgroup='legend', showlegend= c == 0),
                row = c + 1, col = 1
            )
            
        for i in range(len(change_timestamps) - 1):
            fig.add_vline(
                x=change_timestamps[i], line_width=3, line_dash="dash", line_color="black",
                row = c+1, col = 1
            )
            
            if c == 0:
                fig.add_vrect(
                    x0=change_timestamps[i], x1=change_timestamps[i+1],
                    fillcolor='white', opacity=0,
                    layer="below", line_width=0,
                    annotation_text=f'<b>{activities[i]}</b>', annotation_font_size=13, annotation_font_color='black', annotation_font_family='Courier',
                    annotation_position='bottom', annotation_xanchor='center', 
                    annotation_yshift=-25, annotation_bordercolor='black',
                    row = c+1, col = 1
                )
        
    for ax in fig['layout']:
        if ax[:5] == 'yaxis':
            fig['layout'][ax]['gridcolor'] = 'darkgrey'
            fig['layout'][ax]['gridwidth'] = 1
            fig['layout'][ax]['zeroline'] = True
            fig['layout'][ax]['zerolinecolor'] = 'black'
            fig['layout'][ax]['zerolinewidth'] = 2
        elif ax[:5] == 'xaxis':
            fig['layout'][ax]['gridcolor'] = 'darkgrey'
            fig['layout'][ax]['gridwidth'] = 1
         
    source_device = 'Smartphone' if execution.split('_')[-1] == 'sp' else 'Smartwatch'
    fig.update_layout(#height=600, width=1200,
        title={
            'text': f'{source_device} accelerometer (top) and gyroscope (bottom) samples',
            'y': 0.90,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 22
        },
        plot_bgcolor = 'rgb(255,255,255)', showlegend=True)
    fig.update_xaxes(dtick=500, tickformat='%H:%M:%S.%2f')
    fig.update_xaxes(title_text='Timestamp', row=2, col=1)
    fig.update_yaxes(title_text='Intensity (m/s<sup>2</sup>)', row=1, col=1)
    fig.update_yaxes(title_text='Angular velocity (º/s)', row=2, col=1)
    fig.update_layout(margin=dict(l=20, r=20, t=100, b=20), font_family='Helvetica')
    return fig


def plot_orientation_stats(executions_info):
    '''
    Generates an interactive plot counting the different phone orientations in the executions.
    
    Args:
        executions_info (`pandas.DataFrame`): DataFrame with the information of the executions. See: `data_loading.load_executions_info()`
        
    Returns:
        figure (`plotly.graph_objs.Figure`): Interactive plot
    '''
    df = executions_info['orientation'].value_counts()
    fig = go.Figure(data=[go.Bar(x=df.index, y=df.values, text=df.values)])
    fig.update_layout(
        width=500,
        title={
            'text': 'Phone orientation in pocket',
            'y': 0.90,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 22
        }, 
        xaxis_title='Orientation', 
        yaxis_title='Count'
    )

    return fig


def plot_turn_direction_stats(executions_info): 
    '''
    Generates an interactive plot counting the turning direction (right or left) of the `first_turn` and `second_turn`
    
    Args:
        executions_info (`pandas.DataFrame`): DataFrame with the information of the executions. See: `data_loading.load_executions_info()`
        
    Returns:
        figure (`plotly.graph_objs.Figure`): Interactive plot
    '''

    first = executions_info['first_turn'].value_counts()
    second = executions_info['second_turn'].value_counts()

    fig = go.Figure([
        go.Bar(x=first.index, y=first.values, text=first.values, name='first_turn'),
        go.Bar(x=second.index, y=second.values, text=second.values, name='second_turn')
    ])

    fig.update_layout(
        width=500,
        title={
            'text': 'Turn direction',
            'y': 0.90,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 22
        }, 
        xaxis_title='Direction', 
        yaxis_title='Count'
    )
    
    return fig


def plot_turn_direction_combined_stats(executions_info):
    '''
    Generates an interactive plot counting the turning direction of the `first_turn` and `second_turn` combined.
    
    Args:
        executions_info (`pandas.DataFrame`): DataFrame with the information of the executions. See: `data_loading.load_executions_info()`
        
    Returns:
        figure (`plotly.graph_objs.Figure`): Interactive plot
    '''

    df = executions_info[['first_turn', 'second_turn']].value_counts()

    df = df.reset_index()
    df['comb'] = df['first_turn'] + df['second_turn']
    fig = go.Figure([go.Bar(x=df['comb'], y=df[0], text=df[0])])
    fig.update_layout(
        width=500,
        title={
            'text': 'Turn direction combined',
            'y': 0.90,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 22
        }, 
        xaxis_title='Direction', 
        yaxis_title='Count'
    )

    return fig