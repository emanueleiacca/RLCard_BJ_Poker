import sys
sys.path.append('C:/Users/emanu/THESIS_BJ_POKER_REINFORCEMENTLEARNING/')
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
        savedir="experiments/dmc_result_final",
        save_interval=1,
        total_frames=10000000,
        exp_epsilon=0.025,
        batch_size=35,
        unroll_length=30,
        num_buffers=50,
        num_threads=4,
        max_grad_norm=25,
        learning_rate=9.064264760911147e-04,
        alpha=0.95,
        momentum=0,
        epsilon=1e-8 
    )

    # Start training
    trainer.start()
