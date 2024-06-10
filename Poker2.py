''' A toy example of playing against pretrianed AI on Leduc Hold'em
'''
import sys
sys.path.append('C:/Users/emanu/Thesis/Thesis_BJ_Poker_ReinforcementLearning/') 

from agents import RandomAgent
from models import limitholdem_rule_models
from models.limitholdem_rule_models import LimitholdemRuleAgentV1
from agents import LimitholdemHumanAgent as HumanAgent
from utils import print_card
from envs import make
# Make environment
env = make('limit-holdem')

human_agent = HumanAgent(env.num_actions)
agent = LimitholdemRuleAgentV1()
# random_agent = RandomAgent(num_actions=env.num_actions)

env.set_agents([human_agent, agent])


print(">> Leduc Hold'em pre-trained model")

while (True):
    print(">> Start a new game")
    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    print(state)
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            if 'current_player' in state:
                current_player = state['current_player']
            else:
                # Handle the case where 'current_player' key is missing
                current_player = None  # Or any default value you prefer

        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     CFR Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    inputs = input("Press any key to continue, Q to exit\n")
    if inputs.lower() == "q":
      break
print(">> Limit Hold'em random agent")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    if len(trajectories[0]) != 0:
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
    print('=============     Random Agent    ============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    input("Press any key to continue...")
