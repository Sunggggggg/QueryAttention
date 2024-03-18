#!/bin/bash

data_dirs=("Exp4" "Exp3_8queries" "Exp3_no_contra" "Exp3" "Exp2")

for dir in $data_dirs
do
    ckpt_path=output/${dir}/64_8_512/checkpoints/model_current.pth
    python experiment_scripts/eval_realestate10k.py --experiment_name vis_realestate --batch_size 2 --gpus 2 --views 2 --checkpoint_path ${ckpt_path}
done

