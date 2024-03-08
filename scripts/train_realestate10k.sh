#!/bin/bash

python experiment_scripts/train_realestate10k.py --experiment_name realestate_query --batch_size 4 --gpus 2 --checkpoint_path realestate_query/64_3_512/checkpoints/model_current.pth --contra --lpips --depth