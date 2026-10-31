#!/usr/bin/env bash
set -euo pipefail
DATA_PATH="${1:-meu_dataset}"
LANG="${2:-pten}"
python validate_dataset.py "$DATA_PATH/test/$LANG"
python train.py \
  --eval_path "$DATA_PATH/test/$LANG" \
  --model_path "outputs/${LANG}_v1" \
  --do_eval --is_pretrained true --threshold 0.5 \
  --tk2word_prob max --bidirectional_combine_type avg
