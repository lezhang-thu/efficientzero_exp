set -ex

python main.py --env PongNoFrameskip-v4 --case atari --opr test --seed 0 --num_gpus 1 --num_cpus 20 --force \
  --test_episodes 32 \
  --load_model \
  --amp_type 'torch_amp' \
  --model_path 'model_100000.p' \
  --info 'Test'
