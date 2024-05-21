''' Register new environments
'''
import sys
sys.path.append('C:/Users/emanu/rlcard_copy/rlcard/') 
from envs.env import Env
from envs.registration import register, make

register(
    env_id='blackjack',
    entry_point='envs.blackjack:BlackjackEnv',
)

register(
    env_id='limit-holdem',
    entry_point='envs.limitholdem:LimitholdemEnv',
)

register(
    env_id='no-limit-holdem',
    entry_point='envs.nolimitholdem:NolimitholdemEnv',
)


