"""Incremental Leaving-One-Subject-Out script.

Performs the ILOSO evaluation.

**Example**:

    $ python 03_incremental-loso.py 
        --data_dir <PATH_OF_DATA> 
        --reports_dir <PATH_TO_STORE_RECORDS>
        --model <MLP,CNN,LSTM,CNN-LSTM>
        --subject <EVALUATION_SUBJECT>
        --batch_size <BATCH_SIZE>
        --epochs <EPOCHS>
        --splits <SPLITS>
"""

import os
import traceback
import argparse
import gc

import sys
sys.path.append("../../..")

from alive_progress import alive_bar

from libs.chapter3.pipeline.data_grouping import generate_lno_group
from libs.chapter3.pipeline.data_reshapers import get_reshaper
from libs.chapter3.pipeline.training import create_trainer
from libs.chapter3.pipeline.training_report import report_writer
from libs.common.data_loading import load_data
from libs.common.data_grouping import generate_training_and_test_sets
from libs.common.ml import generate_report
from libs.common.utils import set_seed


ACTIVITIES = {"SEATED": 0, "STANDING_UP": 1, "WALKING": 2, "TURNING": 3, "SITTING_DOWN": 4}

BATCH_SIZE = 64
EPOCHS = 50
N_SPLITS = 10


def train_models(data, subjects, test_subjects, model_type, batch_size, epochs, n_splits, reports_dir, testing_mode):
    set_seed()
    writers = {}
    reshaper = get_reshaper(model_type)

    try:
        for test_subject in test_subjects:
            with alive_bar(len(subjects) - 1, dual_line=True, title=f'Evaluating models with {test_subject}', force_tty=True) as progress_bar:
                for n in range(1, len(subjects)):
                    for i in range(n_splits):
                        train_subjects = generate_lno_group(subjects, n, test_subject)
                        
                        progress_bar.text = f'Training {i+1}th model with {n} subjects'
                        for source, (x, y) in data.items():
                            x_train, y_train, x_test, y_test = generate_training_and_test_sets(x, y, train_subjects, [test_subject])
                            if reshaper is not None:
                                x_train = reshaper(x_train)
                                x_test = reshaper(x_test)

                            trainer = create_trainer(model_type, source, batch_size, epochs)
                            trainer(x_train, y_train, verbose=0)
                            model, training_time = trainer(x_train, y_train, verbose=0)
                            y_pred = model.predict(x_test, verbose=0)

                            report = generate_report(y_test, y_pred, ACTIVITIES.keys())
                            report['training time'] = training_time

                            if not testing_mode:
                                if not source in writers:
                                    writers[source] = report_writer(os.path.join(reports_dir, f'{0}_models.csv'.format(f'{model_type}_{source}')))
                                writers[source](test_subject, n, i+1, report)
                            
                            del model
                            del x
                            del y
                            del x_train
                            del y_train
                            del x_test
                            del y_test
                    gc.collect()
                    progress_bar()
    except:
        with open('failures.txt', 'a') as file:
            file.write(f'Exception in test_subject:{test_subject}, n_training_subjects:{n}, model:{model_type}, iteration: {i}\n')
            traceback.print_exc(file=file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', type=str, required=True)
    parser.add_argument('--reports_dir', help='directory to store reports', type=str, required=True)
    parser.add_argument('--model', help='model to use for evaluation', type=str, choices=['mlp', 'lstm', 'cnn', 'cnn-lstm'])
    parser.add_argument('--subject', help='evaluate only with specified subject', type=int)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', help='training epochs', type=int, default=EPOCHS)
    parser.add_argument('--splits', help='models trained for each case', type=int, default=N_SPLITS)
    parser.add_argument('--testing_script', help='Testing the script. Results not stored', action='store_true')
    args = parser.parse_args()

    use_raw_data = args.model != 'mlp'

    x_sp, y_sp = load_data(args.data_dir, 'sp', use_raw_data, ACTIVITIES)
    x_sw, y_sw = load_data(args.data_dir, 'sw', use_raw_data, ACTIVITIES)
    x_fused, y_fused = load_data(args.data_dir, 'fused', use_raw_data, ACTIVITIES)
    data = {
        'sp': (x_sp, y_sp),
        'sw': (x_sw, y_sw),
	    'fused': (x_fused, y_fused)
    }
    subjects = list(x_sp.keys())
    test_subjects = [subjects[args.subject - 1]] if args.subject else subjects
    train_models(data, subjects, test_subjects, args.model, args.batch_size, args.epochs, args.splits, args.reports_dir, args.testing_script)  
    