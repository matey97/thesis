# StanWiFi Dataset

This file contains the instructions to download and setup the data of the StanWiFi Dataset, collected in:

> S. Yousefi, H. Narui, S. Dayal, S. Ermon and S. Valaee, "A Survey on Behavior Recognition Using WiFi Channel State Information," in IEEE Communications Magazine, vol. 55, no. 10, pp. 98-104, Oct. 2017, doi: 10.1109/MCOM.2017.1700082.

## Download

The StanWiFi dataset can be downloaded from the following [GitHub repository](https://github.com/ermongroup/Wifi_Activity_Recognition).

## Setup

1. Unzip the dataset into the root of this directory.
2. Download the scripts [`cross_vali_data_convert_merge.py`](https://github.com/ermongroup/Wifi_Activity_Recognition/blob/master/cross_vali_data_convert_merge.py) and [`cross_vali_input_data.py`](https://github.com/ermongroup/Wifi_Activity_Recognition/blob/master/cross_vali_input_data.py) and place them into this directory.
3. Run `python cross_vali_data_convert_merge.py`

## Processing

Run the script `libs/chapter5/pipeline/01.2_stanwifi-processing.py` to process the dataset.