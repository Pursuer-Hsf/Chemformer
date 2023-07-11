#!/bin/bash

python setup.py develop
nohup python -m molbart.fine_tune \
  --dataset uspto_50 \
  --data_path test_data/seq-to-seq_datasets/uspto_50.pickle \
  --model_path none \
  --task backward_prediction \
  --epochs 100 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 \
  --gpus 2\
  --d_model 256 \
  --num_layers 2 \
  --num_heads 4 \
  --d_feedforward 1024 \
  >logs/test_train_selfies_$(date +%y%m%d_%H%M%S).log 2>&1 &
