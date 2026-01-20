python vla-scripts/finetune_libero_demo.py \
  --data_root_dir datasets/modified_libero_rlds \
  --task_suites libero_spatial \
  --val_frequency 50 \
  --val_episodes 10 \
  --num_videos 5 \
  --train_rollout_frequency 0
