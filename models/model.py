import torch.nn as nn

class Model(nn.Module):
    ''' The base model class
    '''

    def __init__(self):
        ''' Initialize the model
        '''
        super(Model, self).__init__()
        # Define your model architecture here

    def forward(self, x):
        ''' Define the forward pass of the model
        '''
        # Implement the forward pass logic here
        pass

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        raise NotImplementedError
