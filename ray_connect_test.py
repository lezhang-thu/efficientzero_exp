import ray
import numpy as np

ray.init(address='auto')

import time


@ray.remote
def f():
    time.sleep(0.01)
    return ray._private.services.get_node_ip_address()


@ray.remote
class ReplayBuffer(object):
    def __init__(self, config=None):
        # debug - start
        print("run within RelayBuffer __init__")
        # debug - end
        self.config = config
        self.batch_size = -1
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = -1
        self.transition_top = -1
        self.clear_time = 0

    def get_batch_size(self):
        return self.batch_size


# Get a list of the IP addresses of the nodes that have joined the cluster.
x = set(ray.get([f.remote() for _ in range(1000)]))
print(x)
#exit(0)


@ray.remote
def g(x, idx):
    time.sleep(0.01)
    print(np.asarray(x) + idx)


x = [0, 1, 2, 3]
y = [g.remote(x, idx) for idx in range(10)]
del x[:]
ray.get(y)

exit(0)

replay_buffer = ReplayBuffer.remote(None)
print(ray.get([replay_buffer.get_batch_size.remote()]))
