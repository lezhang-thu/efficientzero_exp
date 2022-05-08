import ray
import time

from core.test import _test
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage, QueueStorage
from core.selfplay_worker import DataWorker
from core.reanalyze_worker import BatchWorker_CPU
from core.parameter_server import ParameterServer, GradActor


def _train(replay_buffer, shared_storage, mcts_storage, config):
    # set augmentation tools
    if config.use_augmentation:
        config.set_transforms()

    ps = ParameterServer.remote(config)
    grad_actors = [
        GradActor.remote(config, replay_buffer, mcts_storage) for _ in range(2)
    ]
    current_weights = ps.get_weights.remote()
    # sync weights
    shared_storage.set_weights.remote(current_weights)

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_total_len.remote()) >=
               config.start_transitions):
        time.sleep(1)
    print('Begin training...')
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    gradients = {}
    for single_actor in grad_actors:
        gradients[single_actor.compute_gradients.remote(
            current_weights)] = single_actor

    step_count = 0
    while step_count < config.training_steps + config.last_steps:
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        single_actor = gradients.pop(ready_gradient_id)

        # Compute and apply gradients
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id],
                                                    step_count=step_count)
        gradients[single_actor.compute_gradients.remote(
            current_weights)] = single_actor

        shared_storage.incr_counter.remote()
        step_count += 1
        print("step_count: {}".format(step_count))
        # remove data if the replay buffer is full
        if step_count % 1000 == 0:
            replay_buffer.remove_to_fit.remote()

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(current_weights)

        # save model
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir,
                                      'model_{}.p'.format(step_count))
            torch.save(ray.get(current_weights), model_path)

    time.sleep(30)
    return ray.get(ps.get_weights.remote())


def train(config):
    print(ray.cluster_resources())
    storage = SharedStorage.remote(config)
    # prepare the batch and mcts context storage
    mcts_storage = QueueStorage(18, 25)
    replay_buffer = ReplayBuffer.remote(config)
    print('*' * 20)
    print("Create replay_buffer success!")
    print('*' * 20)

    workers = []
    cpu_workers = [
        BatchWorker_CPU.remote(idx, replay_buffer, storage, mcts_storage,
                               config) for idx in range(config.cpu_actor)
    ]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    config.num_actors = 2
    print("self play: {}".format(config.num_actors))
    # self-play workers
    data_workers = [
        DataWorker.options(num_gpus=0.6).remote(rank, replay_buffer, storage,
                                                config)
        for rank in range(0, config.num_actors // 2)
    ]
    data_workers += [
        DataWorker.options(num_gpus=0.6).remote(rank, replay_buffer, storage,
                                                config)
        for rank in range(config.num_actors // 2, config.num_actors)
    ]
    workers += [worker.run.remote() for worker in data_workers]
    # test workers
    workers += [_test.options(num_gpus=0.2).remote(config, storage)]
    # training loop
    final_weights = _train(replay_buffer, storage, mcts_storage, config)

    print('Training over...')
    return final_weights
