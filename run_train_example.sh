#!/usr/bin/env bash

python3 train.py \
  --train_path dataset/portugues-ingles/train/train \
  --eval_path dataset/portugues-ingles/dev/dev \
  --model_path microsoft/mdeberta-v3-base \
  --save_path outputs \
  --log_dir logs \
  --name ingles_portugues \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --bs 8 \
  --do_train \
  --do_eval \
  --is_pretrained False