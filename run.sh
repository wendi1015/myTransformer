CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 train.py \
  --tokenizer_dir ./tokenizer --data_dir ./data --output_dir ./outputs/N_1_and_num_heads_8_d_model_256 \
  --use_rope --N 1 --num_heads 8 --d_model 256 --d_ff 1024 \
  --label_smoothing 0.1 --lr 1e-3 --scheduler cosine --warmup_ratio 0.08 \
  --epochs 20 --batch_size 16 --seed 42 \
  --decode_for_eval greedy --eval_every 1 --eval_samples 128 \
  --eval_max_new_tokens 256 --min_gen_len 40 \
  --early_stop_metric val_loss --early_stop_patience 3