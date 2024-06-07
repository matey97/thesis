import random as py_random
import copy


def pop_random(subjects):
        index = py_random.randint(0, len(subjects) - 1)
        return subjects.pop(index)


def generate_lno_group(subjects, n, test_subject):
    subjects_copy = copy.deepcopy(subjects)
    subjects_copy.remove(test_subject)
    train_subjects = []
    for _ in range(n):
        train_subjects.append(pop_random(subjects_copy))
    del subjects_copy
    return train_subjects