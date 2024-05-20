import sys
sys.path.append('C:/Users/emanu/rlcard_copy - Copia/rlcard/')
from utils.utils import print_card
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
file_path = 'experiments/DQN_TB/Save_Agent/DQN_checkpoint.pth'
params_path = 'optuna_best_trial.json'
agent = DQNAgent.load_model(file_path,params_path)

# Set up agents
env.set_agents([
    agent,
])

#Start the game
print(">> Blackjack AI agent")

while True:
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)

    # Print other players' actions if the agent does not take the final action
    if len(trajectories[0]) != 0:
        final_state = []
        action_record = []
        state = []
        _action_list = []

        for i in range(num_players):
            final_state.append(trajectories[i][-1])
            state.append(final_state[i]['raw_obs'])

        action_record.append(final_state[i]['action_record'])
        for i in range(1, len(action_record) + 1):
            _action_list.insert(0, action_record[-i])

        for pair in _action_list[0]:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Print the dealer's hand and players' hands
    print('===============   Dealer hand   ===============')
    print_card(state[0]['state'][1])

    for i in range(num_players):
        print('===============   Player {} Hand   ==============='.format(i))
        print_card(state[i]['state'][0])

    # Print the game result
    print('===============     Result     ===============')
    for i in range(num_players):
        if payoffs[i] == 1:
            print('Player {} wins {} chips!'.format(i, payoffs[i]))
        elif payoffs[i] == 0:
            print('Player {} ties'.format(i))
        else:
            print('Player {} loses {} chips!'.format(i, -payoffs[i]))
        print('')

    # Wait for user input to continue with the next round
    input("Press any key to continue...")
