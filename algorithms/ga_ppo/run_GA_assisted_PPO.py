from ppo import Memory, PPO
from GA import generate
from features import get_features, get_feature_len
import gym
import gym_simpletetris
import pandas as pd
import torch

############## Hyperparameters ##############
# creating environment
env = gym.make('SimpleTetris-v0', height=20, width=10)
state_dim = get_feature_len()

log_interval = 20           # print avg reward in the interval
max_episodes = 4000        # max training episodes
max_timesteps = 3000         # max timesteps in one episode
update_timestep = 2000     # update policy every n timesteps; batch timesteps
lr = 0.003
gamma = 0.99                # discount factor
K_epochs = 5                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO (standard=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
no_of_runs = 1             # number of runs
NUM_GENERATIONS = 50
POP_SIZE = 20
#############################################

data = {
    'Episode': [],
    'Reward': [],
    'Lines_cleared': [],
    'Run': []
}

def transfer_params(ppo, params, critic_params):
    ppo.policy_old.action_layer.load_state_dict(params)
    ppo.policy.action_layer.load_state_dict(params)
    ppo.policy_old.value_layer.load_state_dict(critic_params)
    ppo.policy.value_layer.load_state_dict(critic_params)
    return ppo


for run in range(1, no_of_runs + 1):
    memory = Memory()
    ppo = PPO(state_dim, lr, gamma, K_epochs, eps_clip)

    print(f"Run {1} \t Beginning genetic algorithm...")
    genetic_params, critic_params = generate(POP_SIZE, NUM_GENERATIONS, state_dim, lr)
    ppo = transfer_params(ppo, genetic_params, critic_params)

    timestep = 0
    cleared_lines = []
    timestep_li = []

    print("Gentic algorithm complete. Beignning PPO algorithm...")
    # training loop
    for i_episode in range(1, max_episodes+1):
        rew_for_plots = 0
        clear = 0
        state = env.reset()
        state_features = get_features(state, 0, [], None, env)

        for t in range(max_timesteps):
            timestep += 1
            state_dict = env.engine.get_valid_final_states(env.engine.shape, env.engine.anchor, env.engine.board)
            act_pairs = [(k, get_features(v[2], v[4], v[5], v[6], env), v[3]) for k, v in state_dict.items()]

            with torch.no_grad():
                actions = ppo.policy_old.act(state_features, memory, act_pairs)

        
            for action in actions:
                _, reward, done, _, info = env.step(action)

            for _, next_state_features, acts in act_pairs:
                if actions == acts:
                    state_features = next_state_features
            
        
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            rew_for_plots += reward
            
            clear = info['lines_cleared']

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
        
            if done:
                break

        timestep_li.append(t)
        cleared_lines.append(clear)

        data['Episode'].append(i_episode)
        data['Reward'].append(rew_for_plots)
        data['Lines_cleared'].append(clear)
        data['Run'].append(run)

        # logging
        if i_episode % log_interval == 0:
            print('Run {}: Episode {} \t max line clear: {} \t max timestep: {}'.format(run, i_episode, max(cleared_lines), max(timestep_li)))
            cleared_lines.clear()
            timestep_li.clear()
    



df = pd.DataFrame(data)
# Save the DataFrame to a CSV file
df.to_csv('results/GA_assisted_PPO.csv', index=False)



