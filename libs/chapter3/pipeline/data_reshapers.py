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


def lstm_reshaper(data):
    return data.transpose(0, 2, 1)


def cnn_lstm_reshaper(data):
    n_windows, n_features, n_timesteps = data.shape
    return data.reshape((n_windows, n_features * n_timesteps)).reshape((n_windows, 5, 10, n_features))


def create_reshaper(reshaper):
    def do_reshape(x):
        return reshaper(x)

    return do_reshape


def get_reshaper(model_type):
    if model_type == 'lstm':
        return create_reshaper(lstm_reshaper)
    if model_type == 'cnn-lstm':
        return create_reshaper(cnn_lstm_reshaper)
    return None