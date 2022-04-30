set -ex

python main.py --env PongNoFrameskip-v4 --case atari --opr train --force \
  --num_gpus 4 --num_cpus 48 --cpu_actor 14 --gpu_actor 20 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1'
