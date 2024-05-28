import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def generate_report(y_test, y_pred, class_names):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        
def report_to_dataframe(test_subject, n, i, report):
    reports_tidy = []

    for metric, value in report.items():
        if isinstance(value, dict):
            for prop in value:
                reports_tidy.append([test_subject, n, i, metric, prop, value[prop]])
        else:
            reports_tidy.append([test_subject, n, i, 'model', metric, value]) 
            
    return pd.DataFrame(reports_tidy, columns=['test_subject', 'n', 'i', 'target', 'metric', 'value'])


def report_writer(file_path):
    first = False
    def writer(test_subject, n, i, report):
        nonlocal first
        df = report_to_dataframe(test_subject, n, i, report)
        if first:
            df.to_csv(file_path, index=False, mode='w')
            first = False
        else:
            df.to_csv(file_path, index=False, header=False, mode='a')
            
    return writer