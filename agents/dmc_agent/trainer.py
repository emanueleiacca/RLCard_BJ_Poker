# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .model import DMCModel
from .pettingzoo_model import DMCModelPettingZoo
from .utils import (
    get_batch,
    create_buffers,
    create_optimizers,
    act,
    log,
)
from .pettingzoo_utils import (
    create_buffers_pettingzoo,
    act_pettingzoo,
)

def compute_loss(logits, targets):
    loss = ((logits - targets)**2).mean()
    return loss

import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def learn(
    position,
    actor_models,
    agent,
    batch,
    optimizer,
    training_device,
    max_grad_norm,
    mean_episode_return_buf,
    lock
):
    """Performs a learning (optimization) step."""
    device = "cuda:"+str(training_device) if training_device != "cpu" else "cpu"
    state = torch.flatten(batch['state'].to(device), 0, 1).float()
    action = torch.flatten(batch['action'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        values = agent.forward(state, action)
        loss = compute_loss(values, target)
        mean_episode_return = torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item()
        stats = {
            'mean_episode_return_'+str(position): mean_episode_return,
            'loss_'+str(position): loss.item(),
        }

        # Log the calculated mean episode return and loss
        logging.info(f"Mean Episode Return at Position {position}: {mean_episode_return}")
        logging.info(f"Loss at Position {position}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_agent(position).load_state_dict(agent.state_dict())

        # Optionally log any other significant event or error
        logging.info(f"Model updated and parameters synchronized for position {position}")

        return stats



class DMCTrainer:    
    """
    Deep Monte-Carlo

    Args:
        env: RLCard environment
        load_model (boolean): Whether loading an existing model
        xpid (string): Experiment id (default: dmc)
        save_interval (int): Time interval (in minutes) at which to save the model
        num_actor_devices (int): The number devices used for simulation
        num_actors (int): Number of actors for each simulation device
        training_device (str): The index of the GPU used for training models, or `cpu`.
        savedir (string): Root dir where experiment data will be saved
        total_frames (int): Total environment frames to train for
        exp_epsilon (float): The prbability for exploration
        batch_size (int): Learner batch size
        unroll_length (int): The unroll length (time dimension)
        num_buffers (int): Number of shared-memory buffers
        num_threads (int): Number learner threads
        max_grad_norm (int): Max norm of gradients
        learning_rate (float): Learning rate
        alpha (float): RMSProp smoothing constant
        momentum (float): RMSProp momentum
        epsilon (float): RMSProp epsilon
    """
    def __init__(
        self,
        env,
        cuda="",
        is_pettingzoo_env=False,
        load_model=False,
        xpid='dmc',
        save_interval=30,
        num_actor_devices=1,
        num_actors=5,
        training_device="0",
        savedir='experiments/dmc_result',
        total_frames=100000000000,
        exp_epsilon=0.01,
        batch_size=32,
        unroll_length=100,
        num_buffers=50,
        num_threads=4,
        max_grad_norm=40,
        learning_rate=0.0001,
        alpha=0.99,
        momentum=0,
        epsilon=0.00001
    ):
        self.env = env

        self.plogger = FileWriter(
            xpid=xpid,
            rootdir=savedir,
        )

        self.checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (savedir, xpid, 'model.tar')))

        self.T = unroll_length
        self.B = batch_size

        self.xpid = xpid
        self.load_model = load_model
        self.savedir = savedir
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.num_actors = num_actors
        self.training_device = training_device
        self.total_frames = total_frames
        self.exp_epsilon = exp_epsilon
        self.num_buffers = num_buffers
        self.num_threads = num_threads
        self.max_grad_norm = max_grad_norm
        self.learning_rate =learning_rate
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon

        self.is_pettingzoo_env = is_pettingzoo_env
        if not self.is_pettingzoo_env:
            self.num_players = self.env.num_players
            self.action_shape = self.env.action_shape
            if self.action_shape[0] == None:  # One-hot encoding
                self.action_shape = [[self.env.num_actions] for _ in range(self.num_players)]

            def model_func(device):
                return DMCModel(
                    self.env.state_shape,
                    self.action_shape,
                    exp_epsilon=self.exp_epsilon,
                    device=str(device),
                )
        else:
            self.num_players = self.env.num_agents

            def model_func(device):
                return DMCModelPettingZoo(
                    self.env,
                    exp_epsilon=self.exp_epsilon,
                    device=device
                )
        self.model_func = model_func

        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.num_players)]

        if cuda == "": # Use CPU
            self.device_iterator = ['cpu']
            self.training_device = "cpu"
        else:
            self.device_iterator = range(num_actor_devices)

    def start(self):
        # Initialize actor models
        models = {}
        for device in self.device_iterator:
            model = self.model_func(device)
            model.share_memory()
            model.eval()
            models[device] = model

        # Initialize buffers
        if not self.is_pettingzoo_env:
            buffers = create_buffers(
                self.T,
                self.num_buffers,
                self.env.state_shape,
                self.action_shape,
                self.device_iterator,
            )
        else:
            buffers = create_buffers_pettingzoo(
                self.T,
                self.num_buffers,
                self.env,
                self.device_iterator,
            )

        # Initialize queues
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = {}
        full_queue = {}
        for device in self.device_iterator:
            _free_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            free_queue[device] = _free_queue
            full_queue[device] = _full_queue

        # Learner model for training
        learner_model = self.model_func(self.training_device)

        # Create optimizers
        optimizers = create_optimizers(
            self.num_players,
            self.learning_rate,
            self.momentum,
            self.epsilon,
            self.alpha,
            learner_model,
        )

        # Stat Keys
        stat_keys = []
        for p in range(self.num_players):
            stat_keys.append('mean_episode_return_'+str(p))
            stat_keys.append('loss_'+str(p))
        frames, stats = 0, {k: 0 for k in stat_keys}

        # Load models if any
        if self.load_model and os.path.exists(self.checkpointpath):
            checkpoint_states = torch.load(
                    self.checkpointpath,
                    map_location="cuda:"+str(self.training_device) if self.training_device != "cpu" else "cpu"
            )
            for p in range(self.num_players):
                learner_model.get_agent(p).load_state_dict(checkpoint_states["model_state_dict"][p])
                optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
                for device in self.device_iterator:
                    models[device].get_agent(p).load_state_dict(learner_model.get_agent(p).state_dict())
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")


        # Starting actor processes
        for device in self.device_iterator:
            num_actors = self.num_actors
            for i in range(self.num_actors):
                actor = ctx.Process(
                    target=act_pettingzoo if self.is_pettingzoo_env else act,
                    args=(i, device, self.T, free_queue[device], full_queue[device], models[device], buffers[device], self.env))
                actor.start()
                actor_processes.append(actor)

        def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal frames, stats
            while frames < self.total_frames:
                batch = get_batch(
                    free_queue[device][position],
                    full_queue[device][position],
                    buffers[device][position],
                    self.B,
                    local_lock
                )
                _stats = learn(
                    position,
                    models,
                    learner_model.get_agent(position),
                    batch,
                    optimizers[position],
                    self.training_device,
                    self.max_grad_norm,
                    self.mean_episode_return_buf,
                    position_lock
                )

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]
                    to_log = dict(frames=frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.plogger.log(to_log)
                    frames += self.T * self.B

        for device in self.device_iterator:
            for m in range(self.num_buffers):
                for p in range(self.num_players):
                    free_queue[device][p].put(m)

        threads = []
        locks = {device: [threading.Lock() for _ in range(self.num_players)] for device in self.device_iterator}
        position_locks = [threading.Lock() for _ in range(self.num_players)]

        for device in self.device_iterator:
            for i in range(self.num_threads):
                for position in range(self.num_players):
                    thread = threading.Thread(
                        target=batch_and_learn,
                        name='batch-and-learn-%d' % i,
                        args=(
                            i,
                            device,
                            position,
                            locks[device][position],
                            position_locks[position])
                        )
                    thread.start()
                    threads.append(thread)

        def checkpoint(frames):
            log.info('Saving checkpoint to %s', self.checkpointpath)
            _agents = learner_model.get_agents()
            torch.save({
                'model_state_dict': [_agent.state_dict() for _agent in _agents],
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                "stats": stats,
                'frames': frames,
            }, self.checkpointpath)

            # Save the weights for evaluation purpose
            for position in range(self.num_players):
                model_weights_dir = os.path.expandvars(os.path.expanduser(
                    '%s/%s/%s' % (self.savedir, self.xpid, str(position)+'_'+str(frames)+'.pth')))
                torch.save(
                    learner_model.get_agent(position),
                    model_weights_dir
                )

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer() - self.save_interval * 60
            while frames < self.total_frames:
                start_frames = frames
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > self.save_interval * 60:
                    checkpoint(frames)
                    last_checkpoint_time = timer()

                end_time = timer()
                fps = (frames - start_frames) / (end_time - start_time)
                log.info(
                    'After %i frames: @ %.1f fps Stats:\n%s',
                    frames,
                    fps,
                    pprint.pformat(stats),
                )
        except KeyboardInterrupt:
            return
        else:
            for thread in threads:
                thread.join()
            log.info('Learning finished after %d frames.', frames)

        checkpoint(frames)
        self.plogger.close()
