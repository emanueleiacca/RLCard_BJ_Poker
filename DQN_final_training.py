import sys
sys.path.append('C:/Users/emanu/Thesis/Thesis_BJ_Poker_ReinforcementLearning/') 
import tensorboard
import torch
import tensorflow as tf
import numpy as np
import importlib
import json
from envs import registration
from envs import make
from agents.poker_dqn import DQNAgent
from utils import calculate_metrics_from_trajectories
from utils import win_rate_function
from utils.utils import (
    tournament,
    reorganize,
    plot_curve,
)
from utils import Logger
#Open json file to retrieve the hyperparameters configuration
with open('optuna_poker.json', 'r') as f:
    best_params = json.load(f)

#Create the environment
env = make("no-limit-holdem")

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
mlp_layers="64,64"
mlp_layers_list = [int(x) for x in mlp_layers.split(',')]

agent2 = DQNAgent(
    replay_memory_size=25000,
    replay_memory_init_size=300,
    update_target_estimator_every=3000,
    discount_factor=0.94,
    epsilon_start=0.8,
    epsilon_end=0.06,
    epsilon_decay_steps=20000,
    batch_size=45,
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=mlp_layers_list, 
    learning_rate=0.0008 ,
)
env.set_agents([agent, agent2])
tb_log_dir = "experiments/DQN_PolerTB/tensorboard_logs"
writer = tf.summary.create_file_writer(tb_log_dir)
all_in_action_count = 0 
player_0_fold_count = 0
player_1_fold_count = 0
total_games = 0
count_fin = 0
count = 0
cumulative_payoff = 0
#Training Loop
#se tutto va bene penso servano 20kk episodi o poco meno
with Logger("experiments/DQN_Poker_TB") as logger:
    for episode in range(5000): #fixed number of episodes #5000/500 sono 3kk episode        
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        first_payoff = payoffs[0]
        cumulative_payoff += (abs(first_payoff)*2)
        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        #print(trajectories)
        for traj in trajectories:
            if traj: 
                last_complex_structure = traj[-1] 
                if isinstance(last_complex_structure, list):
#number of action per game
                    for item in last_complex_structure:

                        if isinstance(item, dict) and 'action_record' in item:
                            last_action_record = item['action_record']
                            total_games += 1
                            # Convert last_complex_structure to a string for easy counting
                            last_complex_structure_str = str(last_action_record)
                            #print(last_complex_structure_str)
                            # Count the occurrences of "("
                            count_fin += last_complex_structure_str.count('(')
                            
                
                #print(f"Number of '(' characters inside last_complex_structure: {count}")
                action_record_found = False
#Counting time that player 0 goes all-in
                for item in last_complex_structure:
                    if isinstance(item, dict) and 'action_record' in item:
                        for action in item['action_record']:
                            last_action_record = item['action_record']
                            action_str = str(action) 
                            if '(0, <Action.ALL_IN: 4>)' in action_str:
                                all_in_action_count += 1
                                break  
#Counting time that player 0 fold
                for item in last_complex_structure:
                    if isinstance(item, dict) and 'action_record' in item:
                        for action in item['action_record']:
                            last_action_record = item['action_record']
                            action_str = str(action) 
                            if '(0, <Action.FOLD: 0>)' in action_str:
                                player_0_fold_count += 1
                                break  
#Counting time that player 1 fold
                for item in last_complex_structure:
                    if isinstance(item, dict) and 'action_record' in item:
                        for action in item['action_record']:
                            last_action_record = item['action_record']
                            action_str = str(action) 
                            if '(1, <Action.FOLD: 0>)' in action_str:
                                player_1_fold_count += 1
                                break  
        avg_reward = tournament(env, 50)[0]
        q_loss_sum = 0  # Initialize sum of Q-Losses
        train_steps = 0  # Initialize number of training steps
        all_in_action_percentage = all_in_action_count / total_games * 100
        player_0_fold_percentage = player_0_fold_count / total_games * 100
        player_1_fold_percentage = player_1_fold_count / total_games * 100
        avg_game_lenght = count_fin / total_games
        average_pot_size = cumulative_payoff / (total_games)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            loss = agent.feed(ts)
            if loss is not None:  # Check if training occurred and loss was returned
                q_loss_sum += loss
                train_steps += 1

        # Evaluate the performance every 50 episodes.
        if episode % 50 == 0:
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
                tf.summary.scalar('Q_Loss', avg_q_loss, step=env.timestep)
                tf.summary.scalar('average_pot_size', average_pot_size, step=env.timestep)
                tf.summary.scalar('all_in_action_percentage', all_in_action_percentage, step=env.timestep)
                tf.summary.scalar('player_0_fold_percentage', player_0_fold_percentage, step=env.timestep)
                tf.summary.scalar('player_1_fold_percentage', player_1_fold_percentage, step=env.timestep)
                tf.summary.scalar('avg_game_lenght', avg_game_lenght, step=env.timestep)

    csv_path, fig_path = logger.csv_path, logger.fig_path

file_path = 'experiments/DQN_Poker_TB/Save_Agent/DQN_checkpoint.pth'
agent.save_model(file_path)
#print("Total occurrences of (0, <Action.ALL_IN: 4>):", all_in_action_count)

'''
DEBUGGING TRAJECTORIES
        for traj in trajectories:
            if traj:  
                last_complex_structure = traj[-1]  
                action_record_found = False

                for item in last_complex_structure:
                    if isinstance(item, dict) and 'action_record' in item:
                        last_action_record = item['action_record']
                        print(f"Last action record for one of the trajectories: {last_action_record}")
                        action_record_found = True
                        break  

                if not action_record_found:
                    print("Action record not found in the last item of the trajectory.")
            else:
                print("Encountered an empty trajectory, skipping to next.")
'''
'''
DEBUGGING METRICS
        if episode % 50 == 0:
            
            print(f"Avg Reward: {avg_reward}")
            print(f"Average Q-Loss: {avg_q_loss if train_steps > 0 else 'N/A'}")
            print(f"Average Pot Size: {average_pot_size}")
            print(f"All-In Action Percentage: {all_in_action_percentage}")
            print(f"Player 0 Fold Percentage: {player_0_fold_percentage}")
            print(f"Player 1 Fold Percentage: {player_1_fold_percentage}")
            print(f"Average Game Length: {avg_game_lenght}")
            print(f"Elo Rating Agent 0: {elo_ratings['agent0']}")
            print(f"Elo Rating Agent 1: {elo_ratings['agent1']}")
            print(f"episode: {total_games}")
'''