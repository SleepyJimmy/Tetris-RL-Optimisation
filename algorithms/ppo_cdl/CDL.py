from ppo import ForwardDynamicsModel
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn

device = 'cuda'

class CuriosityDrivenPPO:
    def __init__(self, obs_size, lr):   
        self.dynamics_model = ForwardDynamicsModel(obs_size)
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()


    def compute_intrinsic_reward(self, current_state, action, shape, actual_next_state):
        """
            input:
                array (int): 10x20 array tetris board
                array (int) : action
                array (tuple (int)) : shape of falling piece
            output:
                int : the intrinsic reward (prediction error of next state prediction and actual next state)
        """
        # preprocess the variables to tensors 
        current_state = torch.tensor(current_state, dtype=torch.float32).view(1, -1).to(device)
        action = torch.tensor(action, dtype=torch.float32).view(1, -1).to(device)
        shape = torch.tensor(shape, dtype=torch.float32).view(1, -1).to(device)
        actual_next_state = torch.tensor(actual_next_state, dtype=torch.float32).to(device)

        # predict the next state
        with torch.no_grad():
            next_state_pred = self.dynamics_model(current_state, action, shape)
        
        # calculate prediction error (higher prediction error, higher intrinsic reward)
        pred_error = self.mse_loss(next_state_pred, actual_next_state).item()
        intrinsic_reward = pred_error

        return intrinsic_reward


    def update_dynamics_model(self, states, actions, shapes, next_states):
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1).to(device)
        shapes = torch.tensor(shapes, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        next_state_pred = self.dynamics_model(states, actions, shapes, is_update=True)

        loss = self.mse_loss(next_states, next_state_pred)

        self.dynamics_optimizer.zero_grad()
        loss.backward()
        self.dynamics_optimizer.step()

