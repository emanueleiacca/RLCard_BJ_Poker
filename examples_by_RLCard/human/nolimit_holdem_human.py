''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''
import sys
sys.path.append('C:/Users/emanu/rlcard_copy/rlcard/') 
from agents import RandomAgent
import models
from agents import NolimitholdemHumanAgent as HumanAgent
from models.model import Model
from utils import print_card
from envs import make
import torch
from agents import DQNAgent
# Load the DQN agent

file_path = 'experiments/nolimit_holdem_dqn_result/Save_Agent/DQN_checkpoint.pth'
params_file_path= 'optuna_poker.json'
model = DQNAgent.load_model(file_path,params_file_path)

# Make environment
env = make('no-limit-holdem')

human_agent = HumanAgent(env.num_actions)
# random_agent = RandomAgent(num_actions=env.num_actions)

# Set the DQN agent as one of the agents in the environment
model = DQN(env.num_actions)
env.set_agents([human_agent, model])

while True:
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     Cards all Players    ===============')
    for hands in env.get_perfect_information()['hand_cards']:
        print_card(hands)

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")