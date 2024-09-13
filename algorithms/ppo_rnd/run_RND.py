import torch
from RND import RNDPPO
from ppo import PPO, Memory
import gym
import gym_simpletetris
from features import get_features, get_feature_len
import itertools
import pandas as pd


"""
    References: https://github.com/openai/random-network-distillation
"""

############## Hyperparameters ##############
# creating environment
env = gym.make('SimpleTetris-v0', height=20, width=10)
obs = env.reset()
obs_size = len(obs.flatten())

state_dim = get_feature_len()
log_interval = 20           # print avg reward in the interval
max_episodes = 4000        # max training episodes
max_timesteps = 3000         # max timesteps in one episode
update_timestep = 2000     # update policy every n timesteps; batch timesteps
lr = 0.003
gamma = 0.99                # discount factor
K_epochs = 5                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
no_of_runs = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#############################################

print('learning rate:',lr)

for run in range(1, no_of_runs + 1):
    memory = Memory()
    ppo = PPO(state_dim, lr, gamma, K_epochs, eps_clip)
    agent = RNDPPO(obs_size, lr)


    timestep = 0
    cleared_lines = []

    # training loop
    for i_episode in range(1, max_episodes+1):
        clear = 0
        state = env.reset()
        state_features = get_features(state, 0, [], None, env)
        rew_for_plots = 0

        for t in range(max_timesteps):
            timestep += 1
            current_state = env.engine.board
            shape = env.engine.shape
            state_dict = env.engine.get_valid_final_states(shape, env.engine.anchor, current_state)
            act_pairs = [(k, get_features(v[2], v[4], v[5], v[6], env), v[3]) for k, v in state_dict.items()]
            
            with torch.no_grad():
                actions, act_index = ppo.policy_old.act(state_features, memory, act_pairs)

            
            for action in actions:
                obs, reward, done, _, info = env.step(action)
            next_state = env.engine.board

            for _, next_state_features, acts in act_pairs:
                if actions == acts:
                    state_features = next_state_features
            
            rew_for_plots += reward

            intrinsic_reward = agent.compute_intrinsic_reward(current_state, act_index, shape, next_state)
            reward = reward + intrinsic_reward

            # appending into memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.shapes.append(list(itertools.chain(*shape)))
            memory.next_states.append(next_state)
            memory.board_state.append(current_state.flatten())

            
            clear = info['lines_cleared']

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                agent.update_dynamics_model(memory)
                memory.clear_memory()
                timestep = 0
        
            if done:
                break
        
        cleared_lines.append(clear)

        # logging
        if i_episode % log_interval == 0:
            print('Run {}: Episode {} \t max line clear: {}'.format(run, i_episode, max(cleared_lines)))
            cleared_lines.clear()

