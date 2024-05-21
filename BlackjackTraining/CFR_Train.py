import sys
sys.path.append('C:/Users/emanu/rlcard_copy - Copia/rlcard/')
import tensorboard
import tensorflow as tf
import torch
import numpy as np
import importlib
import json
from envs import registration
from envs import make
from agents import RandomAgent
from agents import CFRAgent
from utils import calculate_metrics_from_trajectories
from utils import win_rate_function
from utils.utils import (
    tournament,
    reorganize,
    plot_curve,
)
from utils import Logger

#Create the environment
env = make(
        'blackjack',
        config={
            'allow_step_back': True,
        }
    )
eval_env = make(
    'blackjack'
)
   
agent = CFRAgent(
    env,
    "cfr_model"
)


env.set_agents([agent]) 
eval_env.set_agents([agent])

tb_log_dir = "experiments/CFR_TB/tensorboard_logs"
writer = tf.summary.create_file_writer(tb_log_dir)

with Logger("experiments/leduc_holdem_cfr_result") as logger:
    for episode in range(100000):  # 100k training episodes
        action_counts = {action: 0 for action in range(env.num_actions)}
        agent.train()
        trajectories, payoffs = env.run(is_training=False)
        trajectories = reorganize(trajectories, payoffs)
        avg_reward = tournament(eval_env, 500)[0]        
        win_rate = win_rate_function(eval_env, 500)
        avg_score, bust_rate = calculate_metrics_from_trajectories(trajectories)
        for ts in trajectories[0]:
            action = ts[1] 
            action_counts[action] += 1

        # Log every 500 episodes
        if episode % 500 == 0:
            logger.log_performance(env.timestep, avg_reward)
            with writer.as_default():
                tf.summary.scalar('average_reward', avg_reward, step=episode)
                tf.summary.scalar('win_rate', win_rate, step=episode)
                tf.summary.scalar('bust_rate', bust_rate, step=episode)
                tf.summary.scalar('avg_score', avg_score, step=episode)

        # Save model every 1000 episodes
        if episode % 1000 == 0:
            agent.save()

    # Save final model
    file_path = 'experiments/CFR_TB/Save_Agent/CFR_checkpoint.pth'
    agent.save_model(file_path)
