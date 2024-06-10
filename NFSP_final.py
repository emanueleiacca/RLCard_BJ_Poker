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
from agents import NFSPAgent, RandomAgent
from utils import calculate_metrics_from_trajectories
from utils import win_rate_function
from utils.utils import (
    tournament,
    reorganize,
    plot_curve,
)
from utils import Logger
#Open json file to retrieve the hyperparameters configuration
with open('optuna_poker_nfsp.json', 'r') as f:
    best_params = json.load(f)

mlp_layers = [int(layer) for layer in best_params['mlp_layers'].split(',') if layer.strip().isdigit()]

#Create the environment
env = make("limit-holdem")
num_players = env.num_players
state_shape = env.state_shape[0]
#print("state_shape", state_shape)
#DQN agent's Parameters
agent = NFSPAgent(
    num_actions=env.num_actions,
    state_shape=state_shape,
    anticipatory_param=best_params['anticipatory_param'],
    hidden_layers_sizes= mlp_layers,
    q_mlp_layers = mlp_layers,
    batch_size=best_params['batch_size'],
    q_discount_factor=best_params['q_discount_factor'],
    q_epsilon_decay_steps=best_params['q_epsilon_decay_steps'],
    q_epsilon_end=best_params['q_epsilon_end'],
    q_epsilon_start=best_params['q_epsilon_start'],
    q_replay_memory_init_size=best_params['q_replay_memory_init_size'],
    q_replay_memory_size=best_params['q_replay_memory_size'],
    q_update_target_estimator_every=best_params['q_update_target_estimator_every'],
    reservoir_buffer_capacity=best_params['reservoir_buffer_capacity'],
    rl_learning_rate=best_params['rl_learning_rate'],
    sl_learning_rate=best_params['sl_learning_rate'],
)
agents = [agent]
for _ in range(1, env.num_players):
    agents.append(RandomAgent(num_actions=env.num_actions))
env.set_agents(agents)
tb_log_dir = "experiments/NFSP_TB/tensorboard_logs"
writer = tf.summary.create_file_writer(tb_log_dir)
all_in_action_count = 0 
total_games = 0
count_fin = 0
count = 0
cumulative_payoff = 0
# Initialize fold counts for players
player_0_fold_count = 0
player_1_fold_count = 0

# Track the number of games processed
game_count = 0
game_count_po = 0
#Training Loop
#se tutto va bene penso servano 20kk episodi o poco meno
with Logger("experiments/NFSP_TB") as logger:
    for episode in range(10000): #fixed number of episodes #5000/500 sono 3kk episode   
        agents[0].sample_episode_policy()        
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        first_payoff = payoffs[0]
        #cumulative_payoff += (abs(first_payoff)*2)
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
                            #print(total_games)
                            # Convert last_complex_structure to a string for easy counting
                            last_complex_structure_str = str(last_action_record)
                            #print(last_complex_structure_str)
                            # Count the occurrences of "("
                            count_fin += last_complex_structure_str.count('(')
                            #print("count_fin",count_fin)
                            cumulative_payoff += (abs(first_payoff)*2)
                            #print(cumulative_payoff)
                            game_count += 1
                            game_count_po += 2
                            #print("game_count",game_count)
                            #print(f"Number of '(' characters inside last_complex_structure: {count_fin}")
                            action_record_found = False
                            for item in last_complex_structure:
                                if isinstance(item, dict) and 'action_record' in item:
                                    last_action_record = item['action_record']
                                    player_0_folded_in_game = False
                                    player_1_folded_in_game = False
                                #Counting time that player 0 fold
                                    for action in last_action_record:
                                        if action == (0, 'fold') and not player_0_folded_in_game:
                                            player_0_fold_count += 1
                                            player_0_folded_in_game = True  # Mark that player 0 folded in this game
                                 #Counting time that player 1 fold
                                        elif action == (1, 'fold') and not player_1_folded_in_game:
                                            player_1_fold_count += 1
                                            player_1_folded_in_game = True  # Mark that player 1 folded in this game
                                            break  # Stop processing further actions for player 1 after a fold
                                    
                                    # Debug information to track the progress
                                    #print(f"Game {game_count} processed, Player 0 fold count: {player_0_fold_count}, Player 1 fold count: {player_1_fold_count}")
        
        avg_reward = tournament(env, 1000)[0]
        q_loss_sum = 0  # Initialize sum of Q-Losses
        train_steps = 0  # Initialize number of training steps
        player_0_fold_percentage = player_0_fold_count / (game_count_po) * 100
        player_1_fold_percentage = player_1_fold_count / (game_count_po) * 100
        avg_game_lenght = count_fin / game_count
        average_pot_size = cumulative_payoff / (game_count)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            sl_loss = agent.feed(ts)
            
            # Debugging information to check the output of feed method
            #print(f"Trajectory step: {ts}, Loss: {sl_loss}")
            
            if sl_loss is not None:  # Check if training occurred and loss was returned
                q_loss_sum += sl_loss
                train_steps += 1
                
                # Debugging information to check the accumulation
                #print(f"Updated q_loss_sum: {q_loss_sum}, Updated train_steps: {train_steps}")

        # Calculate average Q loss
        avg_q_loss = q_loss_sum / train_steps if train_steps > 0 else 0

        # Final debugging information
        #print(f"Final avg_q_loss: {avg_q_loss}")
        #print(f"Final q_loss_sum: {q_loss_sum}, Final train_steps: {train_steps}")
        # Evaluate the performance every 50 episodes.
        if episode % 1000 == 0:
            logger.log_performance(
                env.timestep,
                avg_reward
            )
        #print(f"Train steps: {train_steps}") #Debug for Q-Loss
            #print(f"Logging Q-Loss: {avg_q_loss}") #Debug for Q-Loss
            # Log metrics to TensorBoard
        with writer.as_default():

            #print("Writing average_reward to TensorBoard")
            tf.summary.scalar('average_reward', avg_reward, step=env.timestep)
            #print("Writing average_pot_size to TensorBoard")
            tf.summary.scalar('average_pot_size', average_pot_size, step=env.timestep)
            #print("Writing player_0_fold_percentage to TensorBoard")
            tf.summary.scalar('player_0_fold_percentage', player_0_fold_percentage, step=env.timestep)
            #print("Writing player_1_fold_percentage to TensorBoard")
            tf.summary.scalar('player_1_fold_percentage', player_1_fold_percentage, step=env.timestep)
            #print("Writing avg_game_length to TensorBoard")
            tf.summary.scalar('avg_game_length', avg_game_lenght, step=env.timestep)
            #print(f"Logging Q-Loss: {avg_q_loss}") #Debug for Q-Loss
            #print("Writing Q_Loss to TensorBoard")
            tf.summary.scalar('Q_Loss', avg_q_loss, step=env.timestep)

        writer.flush()  # Ensure that all data is written
    csv_path, fig_path = logger.csv_path, logger.fig_path
print("Total occurrences of (0, <Action.ALL_IN: 4>):", player_0_fold_percentage)
save_path = 'experiments/NFSP_TB/Save_Agent/NFSP_checkpoint.pth'
torch.save(agent, save_path)
print('Model saved in', save_path)
