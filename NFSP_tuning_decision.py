#RL card env, agent and utilities
import sys
sys.path.append('C:/Users/emanu/Thesis/Thesis_BJ_Poker_ReinforcementLearning/') 
from envs import make
from agents import NFSPAgent
from models.limitholdem_rule_models import LimitholdemRuleAgentV1
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
    plot_curve_poker,
)


# Create the environment
env = make("limit-holdem", config={'seed': 0})

def objective(trial):
    # Suggest values for the hyperparameters
    reservoir_buffer_capacity = trial.suggest_int('reservoir_buffer_capacity', 10000, 30000, step=5000)
    anticipatory_param = trial.suggest_float('anticipatory_param', 0.05, 0.2, step=0.05)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    rl_learning_rate = trial.suggest_float('rl_learning_rate', 0.001, 0.1, log=True)
    sl_learning_rate = trial.suggest_float('sl_learning_rate', 0.001, 0.01, log=True)
    min_buffer_size_to_learn = trial.suggest_int('min_buffer_size_to_learn', 50, 500, step=50)
    q_replay_memory_size = trial.suggest_int('q_replay_memory_size', 10000, 30000, step=5000)
    q_replay_memory_init_size = trial.suggest_int('q_replay_memory_init_size', 50, 500, step=50)
    q_update_target_estimator_every = trial.suggest_int('q_update_target_estimator_every', 500, 2000, step=500)
    q_discount_factor = trial.suggest_float('q_discount_factor', 0.9, 0.99, step=0.01)
    q_epsilon_start = trial.suggest_float('q_epsilon_start', 0.01, 0.1, step=0.01)
    q_epsilon_end = trial.suggest_float('q_epsilon_end', 0, 0.01, step=0.001)
    q_epsilon_decay_steps = trial.suggest_int('q_epsilon_decay_steps', int(1e5), int(1e6), step=int(1e5))

    mlp_layer_sizes = ['16,16', '32,32', '64,64', '128,128', '256,256', '512,512', '1024,1024']
    mlp_layers_tuple = trial.suggest_categorical('mlp_layers', mlp_layer_sizes) 
    mlp_layers = [int(x) for x in mlp_layers_tuple.split(',')]

    q_mlp_layer_sizes = ['16,16', '32,32', '64,64', '128,128', '256,256', '512,512', '1024,1024']
    q_mlp_layers_tuple = trial.suggest_categorical('mlp_layers', q_mlp_layer_sizes)
    q_mlp_layers = [int(x) for x in q_mlp_layers_tuple.split(',')] 
    num_players = env.num_players
    #print(num_players)
    agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            q_mlp_layers=q_mlp_layers,
            hidden_layers_sizes= mlp_layers,
            reservoir_buffer_capacity=reservoir_buffer_capacity,
            anticipatory_param=anticipatory_param,
            batch_size=batch_size,
            rl_learning_rate=rl_learning_rate,
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
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(LimitholdemRuleAgentV1())
    env.set_agents(agents)


    print(agents)
    # Generate Directory
    trial_dir = f"experiments/Poker_nfsp_result_DecisionBased/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)  # Check if it was correctly generated
    
    # Define a unique path for this trial's performance log
    trial_log_path = os.path.join(trial_dir, "performance.csv")
    
    # Initialize or reset the log file for this trial
    with open(trial_log_path, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'avg_reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Training Loop
    with Logger(f"experiments/Poker_nfsp_result_DecisionBased/trial_{trial.number}") as logger:
        episode_rewards = [] 
        for episode in range(2000):
            agents[0].sample_episode_policy()
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
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    row = {'episode': env.timestep, 'avg_reward': avg_reward}
                    writer.writerow(row)
                    print(f"Written to CSV: {row}")  # Debug print

        last_x_rewards = episode_rewards[-1000:]
        avg_last_x_reward = sum(last_x_rewards) / len(last_x_rewards)

        # Verify the CSV content
        print(f"Verifying CSV content at {trial_log_path}:")  # Debug print
        with open(trial_log_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print(row)  # Debug print to check CSV content

        # Path for the plot
        save_path = os.path.join(trial_dir, "plot_trial.png")
        # Generate plot with avg reward for episode for each trial
        plot_curve_poker(trial_log_path, save_path, "NFSP")

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
fig.write_html("contour_plot_Poker_nfsp_DecisionBased.html")

# Print in terminal the best trial with its hyperparameters
best_trial = study.best_trial
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")

# Saving data in .json to reutilize it later
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
print(best_trial_params)
with open("optuna_poker_nfsp_DecisionBased.json", "w") as best_trial_file:
    best_trial_file.write(best_trial_params)
