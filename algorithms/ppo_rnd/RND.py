import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SHAPE_SIZE = 8

class RandomTargetNet(nn.Module):
    def __init__(self, obs_size):
        super(RandomTargetNet, self).__init__()
        self.fc1 = nn.Linear(obs_size + 1 + SHAPE_SIZE, 64).to(device)
        self.fc1.weight.requires_grad_(False)
        self.fc1.bias.requires_grad_(False)

        self.fc2 = nn.Linear(64, obs_size).to(device) 
        self.fc2.weight.requires_grad_(False)
        self.fc2.bias.requires_grad_(False)


    def forward(self, state, action, shape, is_update=False):
        x = torch.cat([state, action, shape], dim=-1).to(device)
        with torch.no_grad():
            x = torch.relu(self.fc1(x))
            next_state_pred = self.fc2(x)

        if is_update:
            batch_size = next_state_pred.size(0)
            return next_state_pred.view(batch_size, 10, 20)
        else:
            return next_state_pred.view(10, 20)
    

class PredictorNet(nn.Module):
    def __init__(self, obs_size):
        super(PredictorNet, self).__init__()
        self.fc1 = nn.Linear(obs_size + 1 + SHAPE_SIZE, 64).to(device)
        self.fc2 = nn.Linear(64, obs_size).to(device)    

    def forward(self, state, action, shape, is_update=False): 

        x = torch.cat([state, action, shape], dim=-1).to(device)

        x = torch.relu(self.fc1(x))
        next_state_pred = self.fc2(x)

        if is_update:
            batch_size = next_state_pred.size(0)
            return next_state_pred.view(batch_size, 10, 20)
        else:
            return next_state_pred.view(10, 20)



class RNDPPO:
    def __init__(self, obs_size, lr):   
        self.random_target_net = RandomTargetNet(obs_size)
        self.predictor_net = PredictorNet(obs_size)
        self.predictor_net_optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
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
        target_output = self.random_target_net(current_state, action, shape)
        predicted_output = self.predictor_net(current_state, action, shape)


        # calculate prediction error (higher prediction error, higher intrinsic reward)
        pred_error = self.mse_loss(target_output, predicted_output).item()
        intrinsic_reward = pred_error

        return intrinsic_reward


    def update_dynamics_model(self, memory):
        board_state = torch.tensor(memory.board_state, dtype=torch.float32).to(device)
        actions = torch.tensor(memory.actions, dtype=torch.float32).unsqueeze(-1).to(device)
        shapes = torch.tensor(memory.shapes, dtype=torch.float32).to(device)

        next_state_targets = self.random_target_net(board_state, actions, shapes, is_update=True)
        next_state_preds = self.predictor_net(board_state, actions, shapes, is_update=True)
        aux_loss = self.mse_loss(next_state_targets, next_state_preds)

        self.predictor_net_optimizer.zero_grad()
        aux_loss.backward()
        self.predictor_net_optimizer.step()


