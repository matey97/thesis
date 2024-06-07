import numpy as np

from sklearn.metrics import classification_report


def generate_report(y_test, y_pred, class_names):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
