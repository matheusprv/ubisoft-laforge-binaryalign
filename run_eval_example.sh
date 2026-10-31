python3 train.py \
  --eval_path dataset/portugues-ingles/test/test \
  --model_path outputs/ingles_portugues_2 \
  --do_eval --is_pretrained true --threshold 0.5 \
  --tk2word_prob max --bidirectional_combine_type avg
