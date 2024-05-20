import sys
sys.path.append('C:/Users/emanu/rlcard_copy - Copia/rlcard/')
from agents import dqn_agent
import tensorboard
import torch
import tensorflow as tf
import numpy as np
import importlib
import utils
import json
from envs import registration
from envs import make
from agents import DQNAgent
from utils import calculate_metrics_from_trajectories
from utils import win_rate_function
from utils.utils import (
    tournament,
    reorganize,
    plot_curve,
)
from utils import Logger
#Open json file to retrieve the hyperparameters configuration
with open('optuna_best_trial.json', 'r') as f:
    best_params = json.load(f)

#Create the environment
env = make("blackjack")

#DQN agent's Parameters
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[int(layer) for layer in best_params['mlp_layers'].split(',')], # Convert string to list of ints
    replay_memory_size=best_params['replay_memory_size'],
    replay_memory_init_size=best_params['replay_memory_init_size'],
    update_target_estimator_every=best_params['update_target_estimator_every'],
    discount_factor=best_params['discount_factor'],
    epsilon_start=best_params['epsilon_start'],
    epsilon_end=best_params['epsilon_end'],
    epsilon_decay_steps=best_params['epsilon_decay_steps'],
    batch_size=best_params['batch_size'],
    learning_rate=best_params['learning_rate'],
)

env.set_agents([agent]) #We set the DQNAgent as the agent for the env

tb_log_dir = "experiments/DQN_TB_fin/tensorboard_logs"
writer = tf.summary.create_file_writer(tb_log_dir)

#Training Loop
with Logger("experiments/DQN_TB") as logger:
    for episode in range(5000): #fixed number of episodes #5000/500 sono 3kk episode
        action_counts = {action: 0 for action in range(env.num_actions)}
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        avg_reward = tournament(env, 500)[0]
        avg_score, bust_rate = calculate_metrics_from_trajectories(trajectories)
        win_rate = win_rate_function(env, 500)

        q_loss_sum = 0  # Initialize sum of Q-Losses
        train_steps = 0  # Initialize number of training steps

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            action = ts[1]  # Access the action using its index if ts is a list or tuple
            action_counts[action] += 1
            loss = agent.feed(ts)
            if loss is not None:  # Check if training occurred and loss was returned
                q_loss_sum += loss
                train_steps += 1

            
        # Evaluate the performance every 50 episodes.
        if episode % 500 == 0:
            logger.log_performance(
                env.timestep,
                avg_reward
            )
        #print(f"Train steps: {train_steps}") #Debug for Q-Loss
        if train_steps > 0:
            avg_q_loss = q_loss_sum / train_steps
            #print(f"Logging Q-Loss: {avg_q_loss}") #Debug for Q-Loss
            # Log metrics to TensorBoard
            with writer.as_default():
                tf.summary.scalar('average_reward', avg_reward, step=env.timestep)   
                tf.summary.scalar('win_rate', win_rate, step=env.timestep)
                tf.summary.scalar('Bust Rate', bust_rate, step=env.timestep)
                tf.summary.scalar('Average Score', avg_score, step=env.timestep)
                tf.summary.scalar('Q_Loss', avg_q_loss, step=env.timestep)
                for action, count in action_counts.items():
                    tf.summary.scalar(f'Action_{action}_Count', count, step=env.timestep)

    csv_path, fig_path = logger.csv_path, logger.fig_path

file_path = 'experiments/DQN_TB_fin/Save_Agent/DQN_checkpoint.pth'
agent.save_model(file_path)
