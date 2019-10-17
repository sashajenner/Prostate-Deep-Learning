#!/bin/bash

# !!Be careful this will reset all models saved from the cluster!!

cd models/model_001
python3 hyper_params_001.py
cd ../model_002
python3 hyper_params_002.py
cd ../model_003
python3 hyper_params_003.py
cd ../model_004
python3 hyper_params_004.py
cd ../model_005
python3 hyper_params_005.py
