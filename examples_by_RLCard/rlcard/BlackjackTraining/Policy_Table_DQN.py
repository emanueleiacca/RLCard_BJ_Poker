import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('C:/Users/emanu/rlcard_copy - Copia/rlcard/')
from utils.utils import reorganize
from envs import make
from agents import DQNAgent  

# Make environment
num_players = 1
env = make(
    'blackjack',
    config={
        'game_num_players': num_players,
    },
)

# Load the trained DQN agent
file_path = r'C:\Users\emanu\rlcard_copy\rlcard\experiments\DQN_TB\Save_Agent\DQN_checkpoint.pth'
params_path = r'C:\Users\emanu\rlcard_copy\rlcard\optuna_best_trial.json'
agent = DQNAgent.load_model(file_path, params_path)

# Set up agents
env.set_agents([
    agent,
])

# Start the game
print(">> Blackjack AI agent")

# Initialize arrays to store data for plotting
player_sums = []
dealer_cards = []
agent_decisions = []

for i in range(20000):  #Run 20000 games to observe any possible state (a certain "player sum of the 2 card" to a certain "dealer card")
        trajectories, payoffs = env.run(is_training=False)
        trajectories = reorganize(trajectories, payoffs)
        initial_observation = trajectories[0][0][0]['obs'] #from trajectories extract the sum of the two card of the player and the card showed by the dealer

        player_initial_hand_value, dealer_card_value = initial_observation


        first_action_names = []
        #For loop to extract the first action the player did
        for game in trajectories:
            for step in game:
                for action in step:
                    if isinstance(action, dict) and 'action_record' in action:
                        first_action_name = action['action_record'][0][1]  
                        first_action_names.append(first_action_name)
                        break 
        # Map stand and hit action to binary 0, 1
        if first_action_names:  
            agent_decision = 1 if first_action_names[0] == "hit" else 0
        else:
            agent_decision = None 

        #Append value for plotting
        player_sums.append(player_initial_hand_value)
        dealer_cards.append(dealer_card_value)
        agent_decisions.append(agent_decision)

#Plotting 
plt.scatter(player_sums, dealer_cards, c=agent_decisions, cmap='coolwarm')

plt.xlabel('Player Sum')
plt.ylabel('Dealer Card')
plt.title('Agent Policy')
plt.grid(True)
plt.xticks(range(4, 22))
plt.yticks(range(2, 12))

#Label
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='0 (Stand)'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='1 (Hit)')]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
