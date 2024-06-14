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