import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DATA = [
    [0, 0, 0, 0, 0, 6],
    [3, 0, 0, 0, 0, 0, 6],
    [3, 3, 0, 0, 0, 0, 0, 6],
    [3, 3, 3, 0, 0, 0, 0, 0, 6],
    [0, 0, 0, 0, 6],
    [3, 0, 0, 0, 0, 6],
    [3, 3, 0, 0, 0, 0, 6],
    [3, 3, 3, 0, 0, 0, 0, 6],
    [0, 0, 0, 6],
    [3, 0, 0, 0, 6],
    [3, 3, 0, 0, 0, 6],
    [3, 3, 3, 0, 0, 0, 6],
    [0, 0, 6],
    [3, 0, 0, 6],
    [3, 3, 0, 0, 6],
    [3, 3, 3, 0, 0, 6],
    [0, 6],
    [3, 0, 6],
    [3, 3, 0, 6],
    [3, 3, 3, 0, 6],
    [6],
    [3, 6],
    [3, 3, 6],
    [3, 3, 3, 6],
    [1, 6],
    [3, 1, 6],
    [3, 3, 1, 6],
    [3, 3, 3, 1, 6],
    [1, 1, 6],
    [3, 1, 1, 6],
    [3, 3, 1, 1, 6],
    [3, 3, 3, 1, 1, 6],
    [1, 1, 1, 6],
    [3, 1, 1, 1, 6],
    [3, 3, 1, 1, 1, 6],
    [3, 3, 3, 1, 1, 1, 6],
    [1, 1, 1, 1, 6],
    [3, 1, 1, 1, 1, 6],
    [3, 3, 1, 1, 1, 1, 6],
    [3, 3, 3, 1, 1, 1, 1, 6],
    [1, 1, 1, 1, 1, 6],
    [3, 1, 1, 1, 1, 1, 6],
    [3, 3, 1, 1, 1, 1, 1, 6],
    [3, 3, 3, 1, 1, 1, 1, 1, 6]
]

# Convert the list to a dictionary
ACTION_LIST = {i: entry for i, entry in enumerate(DATA)}



class Memory():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.act_pairs = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.act_pairs[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim*2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )


    def forward(self, x, placement):
        raise NotImplementedError


    def get_action_probability(self, state, act_pairs):
        """
            gets action probabilities from the current state comparing all possible next states. 
            Returns actions probabilities and the value of the current state.
        """
        placements = torch.stack([torch.tensor(v, dtype=torch.float32) for k, v, act in act_pairs], dim=0).to(device)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        states = state_tensor.unsqueeze(0).expand(len(act_pairs), -1)
        
        act_score = self.action_layer(torch.cat([states, placements], dim=-1))
        V = self.value_layer(states)

        # act_score, V = self.forward(states, placements)
        act_score = act_score.flatten()
        assert act_score.shape == (len(act_pairs),)
        act_prob = F.softmax(act_score, dim=0)
        
        return act_prob, V[0, :]


    # for interacting with environment
    def act(self, state, memory, act_pairs):
        action = None
        action_probs, _ = self.get_action_probability(state, act_pairs)

        dist = Categorical(action_probs)
        act = dist.sample()

        if act.item() in ACTION_LIST:
            action = ACTION_LIST[act.item()]
        else:
            raise ValueError("Action not found in ACTION_LIST")
        memory.states.append(state)
        memory.actions.append(act)
        memory.logprobs.append(dist.log_prob(act))

        for _, v, _ in act_pairs:
            memory.act_pairs.append(v)

        return action
    


    def eval_values(self, states, act_pairs):
        """
            this function should use all placements in memory, but break it up into segments of 44, then repeat the current state 44 times to calculate the action probabilities.
        """     
        # Create a list to store all the placements tensors
        placements_list = []
        for i in range(0, len(act_pairs), 44):
            batch = act_pairs[i:i + 44]
            placements = torch.stack([torch.tensor(placement, dtype=torch.float32) for placement in batch], dim=0).to(device)
            placements_list.append(placements)
        
        # Process states
        state_tensors = [torch.tensor(s, dtype=torch.float32).to(device) for s in states]
        repeated_states_list = [s.unsqueeze(0).expand(44, -1) for s in state_tensors]

        # Initialize lists to store probabilities and values
        all_act_probs = []
        all_values = []

        # Iterate over each chunk of placements and corresponding repeated states
        for placements, states in zip(placements_list, repeated_states_list):
            act_score = self.action_layer(torch.cat([states, placements], dim=-1))
            V = self.value_layer(states)
            act_score = act_score.flatten()
            assert act_score.shape == (44,)
            act_prob = F.softmax(act_score, dim=0)
            
            all_act_probs.append(act_prob)
            all_values.append(V[0, :])

        
        # Concatenate all probabilities and values
        act_probs = torch.stack(all_act_probs)
        values = torch.stack(all_values)

        return act_probs, values


    # for ppo update
    def evaluate(self, state, action, act_pairs):
        action_probs, state_values = self.eval_values(state, act_pairs) # act_probs is tensor containing probabilities of all 44 actions in each current state
        dist = Categorical(action_probs) 

        # the length of action tensor should match the length of the action_probs tensor
        # action_probs tensor would be 44 x n where n is the amount of current states in the batch 
        # 'action' needs to be indices to index the distribution, hence use the dictionary to match the action sequence with the indices.
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_values), dist_entropy


class PPO():
    def __init__(self, state_dim, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = 0.01


        self.policy = ActorCritic(state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    # def update(self, memory, timestep, total_timestep):   
    def update(self, memory):   
        # Monte Carlo estimate of state rewards (can be replaced by General Advantage Estimators)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # convert list to tensor
        old_states = memory.states
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_act_pairs = memory.act_pairs

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_act_pairs)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss (no gradient in advantages)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # MseLoss is for the update of critic, dist_entropy denotes an entropy bonus
            loss = -torch.min(surr1, surr2).mean() + 0.5*self.MseLoss(state_values, rewards).mean() - self.ent_coef*dist_entropy.mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
            self.optimizer.step()

    
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        




    
