import sys
sys.path.append('C:/Users/emanu/rlcard_copy/rlcard/') 
#RL card env, agent and utilities
from envs import make
from agents import DQNAgent
import utils
#Tuning hyperparameters
import optuna
from optuna.pruners import SuccessiveHalvingPruner
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

#Create the environment
env = make("no-limit-holdem",config={
            'seed': 0})

def objective(trial):
    # Sample hyperparameters using Optuna, we use an optimal and wide range
    replay_memory_size = trial.suggest_int('replay_memory_size', 1000, 50000)
    replay_memory_init_size = trial.suggest_int('replay_memory_init_size', 100, 1000)
    update_target_estimator_every = trial.suggest_int('update_target_estimator_every', 100, 5000)
    discount_factor = trial.suggest_float('discount_factor', 0.9, 0.99)
    epsilon_start = trial.suggest_float('epsilon_start', 0.5, 1.0)
    epsilon_end = trial.suggest_float('epsilon_end', 0.01, 0.1)
    epsilon_decay_steps = trial.suggest_int('epsilon_decay_steps', 1000, 50000)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
       
    # Sample MLP layer sizes using predefined combinations
    mlp_layer_sizes = ['16,16', '32,32', '64,64', '128,128', '256,256', '512,512', '1024,1024']
    mlp_layers_tuple = trial.suggest_categorical('mlp_layers', mlp_layer_sizes)
    mlp_layers = [int(x) for x in mlp_layers_tuple.split(',')]

    # Instantiate DQNAgent with sampled hyperparameters
    agent1 = DQNAgent(
        replay_memory_size=replay_memory_size,
        replay_memory_init_size=replay_memory_init_size,
        update_target_estimator_every=update_target_estimator_every,
        discount_factor=discount_factor,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        batch_size=batch_size,
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=mlp_layers,
        learning_rate=learning_rate,
    )
    mlp_layers="128,128"
    mlp_layers_list = [int(x) for x in mlp_layers.split(',')]
    agent2 = DQNAgent(
        replay_memory_size=20000,
        replay_memory_init_size=1000,
        update_target_estimator_every=1000,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=50000,
        batch_size=32,
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=mlp_layers_list, 
        learning_rate=0.0005 ,
    )
    #we pass the DQNAgent to the environment
    env.set_agents([agent1,agent2]) 

    #Generate Directory
    trial_dir = f"experiments/Poker_dqn_result/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True) #Check if it was correctly generated
    
    # Define a unique path for this trial's performance log
    trial_log_path = os.path.join(trial_dir, "performance.csv")
    
    # Initialize or reset the log file for this trial
    with open(trial_log_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'avg_reward'])
        writer.writeheader()

    #Training Loop
    with Logger(f"experiments/Poker_dqn_result/trial_{trial.number}") as logger:
        episode_rewards = [] 
        for episode in range(2000):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            avg_reward = tournament(env, 200)[0] 
            episode_rewards.append(avg_reward)  # Add  the current episode's reward

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent1.feed(ts)
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

        last_x_rewards = episode_rewards[-1000:]  #We use the last 150k episode more or less for evaluation since results may depend from a bad start (high exploration)
        avg_last_x_reward = sum(last_x_rewards) / len(last_x_rewards)
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        #path for the plot
        save_path = os.path.join(trial_dir, "plot_trial.png")
    #Generate plot with avg reward for episode for each trial
    plot_curve(logger.csv_path, save_path, "DQN")

    return avg_last_x_reward  

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize', pruner=SuccessiveHalvingPruner(), sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig = optuna.visualization.plot_slice(study)
fig.show()
fig = optuna.visualization.plot_param_importances(study)
fig.show()
fig = optuna.visualization.plot_contour(study)
# Update the layout to adjust the size since the normal size was unreadable
fig.update_layout(
    autosize=False,
    width=3600,  # Width in pixels
    height=2400   # Height in pixels
)
#Compability problem between plotly and Matplotlib, I had to save the results locally in .html
fig.write_html("contour_plot.html")

#Print in terminal the best trial with his hyperparameters
best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")

#Saving data in .json to reutilize it later
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
print(best_trial_params)
best_trial_file = open("optuna_poker.json", "w")
best_trial_file.write(best_trial_params)
best_trial_file.close()
