import subprocess
import sys
from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from agents.dqn_agent import DQNAgent as DQNAgent
    from agents.nfsp_agent import NFSPAgent as NFSPAgent

from agents.cfr_agent import CFRAgent
from agents.human_agents.limit_holdem_human_agent import HumanAgent as LimitholdemHumanAgent
from agents.human_agents.nolimit_holdem_human_agent import HumanAgent as NolimitholdemHumanAgent
from agents.human_agents.leduc_holdem_human_agent import HumanAgent as LeducholdemHumanAgent
from agents.human_agents.blackjack_human_agent import HumanAgent as BlackjackHumanAgent
from agents.random_agent import RandomAgent
