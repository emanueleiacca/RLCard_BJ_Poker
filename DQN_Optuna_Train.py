#RL card env, agent and utilities
import sys
sys.path.append('C:/Users/emanu/Thesis/Thesis_BJ_Poker_ReinforcementLearning/') 
from envs import make
from agents import NFSPAgent
import utils
#Tuning hyperparameters
import optuna
#Utilities
import numpy as np
import os
import json
import csv
#To plot
import matplotlib as plt
import plotly
#RL Card Utilies
from utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


# Create the environment
env = make("limit-holdem", config={'seed': 0})

def objective(trial):
    # Sample hyperparameters using Optuna
    replay_memory_size = trial.suggest_int('replay_memory_size', 1000, 50000)
    replay_memory_init_size = trial.suggest_int('replay_memory_init_size', 100, 1000)
    update_target_estimator_every = trial.suggest_int('update_target_estimator_every', 100, 5000)
    discount_factor = trial.suggest_float('discount_factor', 0.9, 0.99)
    epsilon_start = trial.suggest_float('epsilon_start', 0.5, 1.0)
    epsilon_end = trial.suggest_float('epsilon_end', 0.01, 0.1)
    epsilon_decay_steps = trial.suggest_int('epsilon_decay_steps', 1000, 50000)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    anticipatory_param = trial.suggest_float('anticipatory_param', 0.1, 0.5)
    sl_learning_rate = trial.suggest_float('sl_learning_rate', 1e-5, 1e-3)
    min_buffer_size_to_learn = trial.suggest_int('min_buffer_size_to_learn', 100, 1000)
    q_replay_memory_size = trial.suggest_int('q_replay_memory_size', 1000, 50000)
    q_replay_memory_init_size = trial.suggest_int('q_replay_memory_init_size', 100, 1000)
    q_update_target_estimator_every = trial.suggest_int('q_update_target_estimator_every', 100, 5000)
    q_discount_factor = trial.suggest_float('q_discount_factor', 0.9, 0.99)
    q_epsilon_start = trial.suggest_float('q_epsilon_start', 0.5, 1.0)
    q_epsilon_end = trial.suggest_float('q_epsilon_end', 0.01, 0.1)
    q_epsilon_decay_steps = trial.suggest_int('q_epsilon_decay_steps', 1000, 50000)

    # Sample MLP layer sizes using predefined combinations
    mlp_layer_sizes = ['16,16', '32,32', '64,64', '128,128', '256,256', '512,512', '1024,1024']
    mlp_layers_tuple = trial.suggest_categorical('mlp_layers', mlp_layer_sizes)
    mlp_layers = [int(x) for x in mlp_layers_tuple.split(',')]

    # Instantiate NFSPAgent with sampled hyperparameters
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers=mlp_layers,
        reservoir_buffer_capacity=replay_memory_size,
        anticipatory_param=anticipatory_param,
        batch_size=batch_size,
        rl_learning_rate=learning_rate,
        sl_learning_rate=sl_learning_rate,
        min_buffer_size_to_learn=min_buffer_size_to_learn,
        q_replay_memory_size=q_replay_memory_size,
        q_replay_memory_init_size=q_replay_memory_init_size,
        q_update_target_estimator_every=q_update_target_estimator_every,
        q_discount_factor=q_discount_factor,
        q_epsilon_start=q_epsilon_start,
        q_epsilon_end=q_epsilon_end,
        q_epsilon_decay_steps=q_epsilon_decay_steps,
    )

    # Set the agent in the environment
    env.set_agents([agent]) 

    # Generate Directory
    trial_dir = f"experiments/Poker_nfsp_result/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)  # Check if it was correctly generated
    
    # Define a unique path for this trial's performance log
    trial_log_path = os.path.join(trial_dir, "performance.csv")
    
    # Initialize or reset the log file for this trial
    with open(trial_log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
        writer.writeheader()

    # Training Loop
    with Logger(f"experiments/Poker_nfsp_result/trial_{trial.number}") as logger:
        episode_rewards = [] 
        for episode in range(2000):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            avg_reward = tournament(env, 200)[0] 
            episode_rewards.append(avg_reward)  # Add  the current episode's reward

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)
            if episode < 500:
                eval_frequency = 100  # Less frequent in early stages
            elif episode < 1500:
                eval_frequency = 75  # Increase frequency as the agent starts forming a more stable policy
            else:
                eval_frequency = 50  # More frequent towards the end for fine-tuning

            # Evaluate the performance.
            if episode % 50 == 0:
                logger.log_performance(env.timestep, avg_reward)

                with open(trial_log_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
                    writer.writerow({'episode': episode, 'avg_reward': avg_reward})

        last_x_rewards = episode_rewards[-1000:]  # We use the last 150k episodes more or less for evaluation since results may depend on a bad start (high exploration)
        avg_last_x_reward = sum(last_x_rewards) / len(last_x_rewards)
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        # Path for the plot
        save_path = os.path.join(trial_dir, "plot_trial.png")
        # Generate plot with avg reward for episode for each trial
        plot_curve(logger.csv_path, save_path, "NFSP")

    return avg_last_x_reward  

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

# Plot optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Plot slice
fig = optuna.visualization.plot_slice(study)
fig.show()

# Plot parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Plot contour
fig = optuna.visualization.plot_contour(study)
# Update the layout to adjust the size since the normal size was unreadable
fig.update_layout(
    autosize=False,
    width=3600,  # Width in pixels
    height=2400   # Height in pixels
)
# Save the contour plot as HTML
fig.write_html("contour_plot.html")

# Print in terminal the best trial with its hyperparameters
best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")

# Saving data in .json to reutilize it later
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
print(best_trial_params)
with open("optuna_poker_nfsp.json", "w") as best_trial_file:
    best_trial_file.write(best_trial_params)
