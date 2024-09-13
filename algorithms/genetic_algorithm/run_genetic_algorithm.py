import gym
import gym_simpletetris
import numpy as np
import random
from features import get_features, gaussian_weight, get_feature_len


##### HYPERPARAMETERS #####
env = gym.make('SimpleTetris-v0', height= 20, width=10)
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
GENERATIONS = 25
NUMBER_OF_RUNS = 1
FEATURE_LENGTH = get_feature_len()
NUM_TIMESTEPS = 3000
###########################

class GeneticAgent():
    def __init__(self):      
        self.weights = {}
        for i in range(FEATURE_LENGTH):
            self.weights[i] = np.random.uniform(-1, 1)


    def get_fitness(self, state_feature):
        score = 0
        for i in range(len(state_feature)):
            score += self.weights[i] * state_feature[i]

        return score
    

    def crossover(self, agent):
        child = GeneticAgent()

        weight_keys = list(self.weights.keys())
        n = len(weight_keys)

        # uniform crossover with 0.5 prob
        for i in range(n):
            child.weights[weight_keys[i]] = self.weights[weight_keys[i]] if random.getrandbits(1) else agent.weights[weight_keys[i]]
        
        # mutation
        for i in range(n):
            child.weights[weight_keys[i]] = gaussian_weight(child.weights[weight_keys[i]]) if random.random() < MUTATION_RATE else child.weights[weight_keys[i]]

        return child
        

    def calculate_action(self, env):
        best_fitness = np.NINF
        best_actions = None

        state_dict = env.engine.get_valid_final_states(env.engine.shape, env.engine.anchor, env.engine.board)
        act_pairs = [(get_features(v[2], v[4], v[5], v[6], env), v[3]) for k, v in state_dict.items()]
        
        for pair in act_pairs:
            actions = pair[1]
            fitness = self.get_fitness(pair[0])
            if best_actions is None or fitness > best_fitness:
                best_fitness = fitness
                best_actions = actions

        return best_actions
    

    def simulate_tetris(self, env, render=False):
        env.reset()
        done = False
        cleared = 0
        total_reward = 0
        
        for _ in range(NUM_TIMESTEPS):
            actions = self.calculate_action(env)
            for action in actions:
                _, reward, done, _, info = env.step(action)
            if render:
                env.render()
            total_reward += reward
            if done:
                break
   
        cleared = info['lines_cleared']
        
        return cleared, total_reward
    

def normalise_to_probabilities(arr):
    min_score = np.min(arr)
    normalised_fitnesses = np.array([score - min_score + 1 for score in arr])
    probs = normalised_fitnesses/normalised_fitnesses.sum()
    return probs


def selection(population, fitness_scores):
    probabilities = normalise_to_probabilities(fitness_scores)
    selected_indices = np.random.choice(range(len(population)), size=len(population), replace=True, p=probabilities)
    selected_agents = [population[i] for i in selected_indices]
    return selected_agents



for run in range(1, NUMBER_OF_RUNS + 1):

    best_agent_weights = None
    best_cleared_lines = -1

    population = [GeneticAgent() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        fitnesses = np.array([agent.simulate_tetris(env) for agent in population])
        cleared_lines = np.array([fitness[0] for fitness in fitnesses])
        total_reward = np.array([fitness[1] for fitness in fitnesses])
               
        
        # pick the best agent
        top_indices = total_reward.argsort()[::-1]
        best_agent_index = top_indices[0]


        # Selection
        selected_agents = selection(population, total_reward)
        
        # crossover and mutation
        new_population = []
        for i in range(len(selected_agents) // 2):
            parent1 = selected_agents[2 * i]
            parent2 = selected_agents[2 * i + 1]
            child1 = parent1.crossover(parent2)
            child2 = parent2.crossover(parent1)
            new_population.extend([child1, child2])
            
        population = new_population
        avg_fitness = np.average(total_reward)

        print(f"Run: {run}\tGeneration {generation + 1} - Highest Line Clear: {max(cleared_lines)} - Average fitness: {avg_fitness}")

