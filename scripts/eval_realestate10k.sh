#!/bin/bash

python experiment_scripts/eval_realestate10k.py --experiment_name vis_realestate --batch_size 2 --gpus 2 --views 2 --checkpoint_path realestate_query_1/64_2_512/checkpoints/model_current.pth
