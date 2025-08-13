python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir /home/jiaqizheng/checkpoints/verl_examples/gsm8k/global_step_120/actor \
  --target_dir /home/jiaqizheng/my_models/Qwen3-0.6B-Tooluse-PPOeopch15
