import numpy as np

from alive_progress import alive_bar
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

from libs.common.utils import RANDOM_SEED


def cross_validation(x, y, model_builder, folds, batch_size, epochs, labels):
    reports = []
    skfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    with alive_bar(folds, title=f'Training models', force_tty=True) as progress_bar:
        for i, (train_index, test_index) in enumerate(skfold.split(x, np.argmax(y, axis=1))):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]

            model = model_builder()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            report = evaluate_model(model, x_test, y_test, labels)
            reports.append(report)
            progress_bar()
    return reports


def build_report(y_test, y_pred, labels):
    y_test = np.argmax(y_test, axis=1)
    pred_probas = np.max(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    inner_labels = [i for i in range(len(labels))]
    mean_proba = [float(np.mean(pred_probas[y_pred == i])) for i in range(len(labels))]
    
    accuracy = accuracy_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred, labels=inner_labels).tolist()
    class_report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0, labels=inner_labels)
    return {
        'accuracy': accuracy,
        'confusion_matrix': cf_matrix,
        'classification_report': class_report,
        'mean_prediction_probabilities': mean_proba
    }


def evaluate_model(model, x_test, y_test, labels):
    y_pred = model.predict(x_test, verbose=0)
    return build_report(y_test, y_pred, labels)