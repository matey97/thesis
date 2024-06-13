import os
import json
import datetime
from dateutil import tz
from tensorflow.keras.utils import set_random_seed


def load_json(data_file_path):
    with open(data_file_path, 'r') as file:
        return json.load(file)
    

def save_json(data, dir_path, file_name):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    with open(os.path.join(dir_path, file_name), 'w') as f:
        json.dump(data, f)
    

def list_folder(path):
    subjects = os.listdir(path)
    subjects.sort()
    return subjects


def list_subjects_folders(path):
    subjects = list_folder(path)
    return list(filter(lambda name : os.path.isdir(os.path.join(path, name)) and name.startswith('s'), subjects))


def datetime_from(execution):
    return datetime.datetime(
        execution.year,
        execution.month,
        execution.day,
        execution.hour, 
        execution.minute,
        execution.second, 
        execution.ms * 1000,
        tzinfo = tz.gettz('Europe/Madrid')
    )


RANDOM_SEED = 5353
def set_seed():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)