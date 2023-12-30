#!/bin/sh
python "python-source/transformers/examples/pytorch/language-modeling/run_clm.py" \
    --model_name_or_path distilgpt2 \
    --train_file "$1" \
    --do_train \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --output_dir "$2"