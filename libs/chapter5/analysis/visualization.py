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
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"


def plot_confusion_matrix(report, prediction_target, labels):
    '''
    Plots the confusion matrix resulting from the evaluation of a machine learning model.

    Args:
        report (dict): Dict containing a classification report from a model evaluation.
        prediction_target (str): String describing what is being classified.
        labels (list[str]): List with the associated classes labels.
    
    Returns:
        (`plotly.Figure`): Interactive Plotly figure.
    '''
    global_accuracy = report['accuracy']
    confusion_matrix = np.array(report['confusion_matrix'])
    cf_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    text_labels = []
    num_classes = len(labels)
    for i in range(num_classes):
        for j in range(num_classes):
            quant = confusion_matrix[i][j]
            acc = cf_matrix[i][j]
            text_labels.append(f'{quant}<br>{acc:.1%}')

    fig = go.Figure(
        go.Heatmap(
            z=cf_matrix, x=labels, y=labels,
            colorscale='blues', showscale=False,
            text=np.asarray(text_labels).reshape(num_classes, num_classes), texttemplate="%{text}", textfont_size=14
        ),
        
    )
    
    fig.add_annotation(text=f'<b>Accuracy: {global_accuracy:.2%}</b>', font_size=20,
                  xref="x domain", yref="y domain",
                  x=0.5, y=1.07, showarrow=False)
    
    fig.update_layout(
        width=800, 
        height=800,
    )
    fig.update_yaxes(title=f"<b>True {prediction_target}</b>", title_font_size=20, autorange="reversed", tickangle=-20)
    fig.update_xaxes(title=f"<b>Predicted {prediction_target}</b>", title_font_size=20, tickangle=-20)
    return fig