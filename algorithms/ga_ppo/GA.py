import numpy as np
import torch
import torch.nn as nn
import gym
import gym_simpletetris
from features import get_features

####### PARAMETERS #######
env = gym.make('SimpleTetris-v0', height=20, width=10)
env.reset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MUTATION_RATE = 0.1
MUTATION_FACTOR = 0.3
#############################


class Memory():
    def __init__(self):
        self.states = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.states[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Network(nn.Module):
    def __init__(self, state_dim):
        super(Network, self).__init__()
        # actor
        self.network = nn.Sequential(
                nn.Linear(state_dim*2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                )


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # critic
        self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                )


def calculate_action(env, network, state_features):
    best_fitness = np.NINF
    best_actions = None
    next_state = None

    state_dict = env.engine.get_valid_final_states(env.engine.shape, env.engine.anchor, env.engine.board)
    act_pairs = [(get_features(v[2], v[4], v[5], v[6], env), v[3]) for k, v in state_dict.items()]

    state_feature_tensor = torch.tensor(state_features, dtype=torch.float32).to(device)
    
    for pair in act_pairs:
        actions = pair[1]
        feature_tensor = torch.tensor(pair[0], dtype=torch.float32).to(device)
        fitness = network.network(torch.cat([state_feature_tensor, feature_tensor]))

        if best_actions is None or fitness > best_fitness:
            best_fitness = fitness
            best_actions = actions
            next_state = pair[0]

    return best_actions, next_state


def selection(pop, fitnesses):
    min_score = np.min(fitnesses)
    normalised_fitnesses = np.array([score - min_score + 1 for score in fitnesses])
    probs = normalised_fitnesses/normalised_fitnesses.sum()

    selected_indices = np.random.choice(range(len(pop)), size=len(pop), replace=True, p=probs)
    new_pop = [pop[i] for i in selected_indices]

    return new_pop



def crossover(pop, in_dim):
    """
        Uniform crossover for a NN regardless of layer size and type (weights or biases).
    """
    new_pop = []

    for i in range(len(pop) // 2):
        child1 = Network(in_dim).to(device)
        child2 = Network(in_dim).to(device)

        parent1 = pop[2 * i]
        parent2 = pop[2 * i + 1]

        for param_a, param_b, param_c1, param_c2 in zip(parent1.parameters(), parent2.parameters(), child1.parameters(), child2.parameters()):
            with torch.no_grad():
                # Apply crossover logic to weights and biases
                for index in np.ndindex(param_a.shape):
                    if np.random.random() > 0.5:
                        param_c1.data[index] = param_b.data[index]
                    else:
                        param_c1.data[index] = param_a.data[index]
                        
                    if np.random.random() > 0.5:
                        param_c2.data[index] = param_b.data[index]
                    else:
                        param_c2.data[index] = param_a.data[index]

        new_pop.extend([child1, child2])
    
    return new_pop


def mutate_multi(pop):
    for model in pop:
        # Iterate over all parameters (weights and biases) of the model
        for param in model.parameters():
            with torch.no_grad():
                # Iterate over each element in the tensor
                for index in np.ndindex(param.shape):
                    if np.random.random() < MUTATION_RATE:
                        noise = torch.randn(1).mul_(MUTATION_FACTOR).to(device)
                        param[index].add_(noise[0])
    return pop




def eval_network(network, critic, memory, optimiser):
    obs = env.reset()

    buffer_size = 200
    done = False
    total_reward = 0
    state_features = get_features(obs, 0, [], None, env)

    for _ in range(3000):        
        memory.states.append(state_features)        
        actions, state_features = calculate_action(env, network, state_features)
        
        for action in actions:
            _, reward, done, _, info = env.step(action)

        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        total_reward += reward

        if len(memory.states) >= buffer_size:
            train_critic(critic, memory, optimiser)
            memory.clear_memory()

        if done:

            break

    return total_reward


def train_critic(critic, memory, optimiser):
    MseLoss = nn.MSELoss()

    rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (0.99 * discounted_reward)
        rewards.insert(0, discounted_reward)
    
    rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
    states_np = np.array(memory.states)
    states = torch.tensor(states_np, dtype=torch.float32).to(device)

    for _ in range(5):
        prediction = critic.network(states)
        loss = MseLoss(rewards, prediction)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()



def generate(pop_size, num_generation, in_dim, lr):
    pop = None
    best_fitness = np.NINF
    fittest_agent = None

    critic_network = Critic(in_dim).to(device)
    optimiser = torch.optim.Adam(critic_network.network.parameters(), lr=lr)
    memory = Memory()


    for generation in range(num_generation):
        if pop is None:
            pop = [Network(in_dim).to(device) for _ in range(pop_size)]
        
        fitnesses = [0] * pop_size
        results = [0] * pop_size

        for i in range(pop_size):
            results[i] = eval_network(pop[i], critic_network, memory, optimiser)

        for i in range(pop_size):
            fitnesses[i] = results[i]

        print(f"Generation {generation + 1} - Max fitness: {max(fitnesses)}")

        fitness_of_best_agent = max(fitnesses)
        if fitness_of_best_agent > best_fitness:
            best_fitness = fitness_of_best_agent
            fittest_agent = pop[np.argmax(fitnesses)]
            fittest_params = fittest_agent.network.state_dict()

        fitnesses = np.array(fitnesses)

        new_pop = selection(pop, fitnesses)
        new_pop = crossover(new_pop, in_dim)
        new_pop = mutate_multi(new_pop)

        pop = new_pop

        critic_params = critic_network.network.state_dict()

    return fittest_params, critic_params


