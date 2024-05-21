import sys
sys.path.append('C:/Users/emanu/rlcard_copy/rlcard/')

import os
import argparse

import rlcard
from agents import (
    DQNAgent,
    RandomAgent,
)
from utils import (
    get_device,
    set_seed,
    tournament,
)
from envs import make
import sys

import torch
from agents import DQNAgent
import json


import json

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        # Load the model parameters
# Load the model parameters
        model_params = torch.load(model_path, map_location=device)
        #print("Model Parameters:", model_params)
    elif os.path.isdir(model_path):  # CFR model
        from agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
        return agent
    else:
        model_path == 'random'
        from agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
        return agent

    # Load Optuna best trial parameters
    with open('optuna_best_trial.json', 'r') as f:
        optuna_params = json.load(f)
    
    # Extract parameters from Optuna best trial
    mlp_layers = optuna_params.get('mlp_layers', [128, 128])
    state_shape = optuna_params.get('state_shape',env.state_shape[0])  

    # Ensure mlp_layers is a list of integers
    if isinstance(mlp_layers, str):
        mlp_layers = [int(layer) for layer in mlp_layers.split(',')]

    # Instantiate a DQNAgent with the loaded parameters, MLP layers, and state shape
    agent = DQNAgent(num_actions=env.num_actions, state_shape=state_shape, mlp_layers=mlp_layers)

    # Set the parameters of the DQNAgent
    agent.__dict__.update(model_params)

    # Set the device
    agent.set_device(device)
    
    return agent



def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = make(args.env, config={'seed': args.seed})
     #print("Environment State Shape:", env.state_shape)

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    print("Tournament Rewards:", rewards)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='blackjack',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'no-limit-holdem',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'experiments/dmc_tuning_result_3/blackjack/0_2003400.pth',
        ],
    )
    #experiments/DQN_TB_fin/Save_Agent/DQN_checkpoint.pth FOR DQN #-0.18484
    #experiments/dmc_tuning_result_3/blackjack/0_2003400.pth FOR DMC # -0.39599
    #
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=100000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)
    args = parser.parse_args()
    print("Models provided:", args.models)
