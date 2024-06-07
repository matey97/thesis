import pandas as pd
        
        
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