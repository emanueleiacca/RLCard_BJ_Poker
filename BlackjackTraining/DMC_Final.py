import sys
sys.path.append('C:/Users/emanu/rlcard_copy - Copia/rlcard/')
import torch
import tensorflow as tf
import numpy as np
from envs import make
from agents import dmc_agent
from agents.dmc_agent import DMCTrainer
env = make("blackjack")
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Initialize DMCTrainer for training
    trainer = DMCTrainer(
        env=env,
        xpid="blackjack",
        savedir="experiments/dmc_result",
        save_interval=1,
        total_frames=1000000,
        exp_epsilon=0.01,
        batch_size=32,
        unroll_length=100,
        num_buffers=50,
        num_threads=4,
        max_grad_norm=40,
        learning_rate=1.064264760911147e-04,
        alpha=0.99,
        momentum=0,
        epsilon=0.00001
    )

    # Start training
    trainer.start()



