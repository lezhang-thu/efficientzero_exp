import os
import ray
import time

from core.replay_buffer import ReplayBuffer


def train(config, summary_writer, model_path=None):
    # debug - start
    print(ray.cluster_resources())
    time.sleep(5)
    replay_buffer = ReplayBuffer.remote(config)
    ray.get([replay_buffer.get_batch_size.remote()])
    exit(0)

    return None, None
