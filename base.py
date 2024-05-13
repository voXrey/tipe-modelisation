import numpy as np
import matplotlib.pyplot as plt
import toml
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from enum import Enum
from collections import namedtuple, deque
import random
from utils import *
import torch.optim as optim
from pathlib import Path
import torch.nn.functional as F
from simulation_env import SimulationEnv
import gymnasium
import math
import random
import matplotlib.pyplot as plt
from itertools import count
import matplotlib

from classes import Position


config_path = Path("config.toml")
config = toml.load(config_path)

match config:
    case {}:
        pass
    case _:
        raise ValueError(f"invalid configuration: {config}")


class Action:
    def __init__(self, x_intensity:float, y_intensity:float):
        self.x_intensity = x_intensity
        self.y_intensity = y_intensity

class Entity:
    def __init__(self, position:Position, radius:float=0, speed:float=0) -> None:
        self.position = position
        self.radius = radius
        self.speed = speed

    def collision(self, entity) -> bool:
        return self.position.distance(entity.position) < max(self.radius, entity.radius)

class Goal(Entity):
    def __init__(self, position:Position, radius:float) -> None:
        super().__init__(position=position, radius=radius)

class Agent(Entity):
    def __init__(self, id:int, position:Position, radius:float, speed:float, goal:Goal) -> None:
        super().__init__(position=position, radius=radius, speed=speed)
        self.id = id
        self.goal = goal
        self.simulation:Simulation = None

    def goal_reached(self) -> bool:
        return self.collision(self.goal)

class Simulation:
    def __init__(self, agents:list[Agent]=[]) -> None:
        self.agents:dict[int,Agent] = dict()
        for agent in agents: self.add_agent(agent)

    def add_agent(self, agent:Agent) -> None:
        agent.simulation = self
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id:int) -> None:
        del self.agents[agent_id]

    def get_agent(self, agent_id:int) -> None|Agent:
        try: return self.agents[agent_id]
        except: return

    def reward(self, state) -> float:
        total_reward = 1
        goal_position = Position(state[0], state[1])
        
        i = 2
        while i+1 < len(state):
            agent_position = Position(
                x = state[i],
                y = state[i+1]
            )
            total_reward += agent_position.distance(goal_position)
            i += 2

        return 1/total_reward
        

    def step(self, agent:Agent, action:Action):
        angle = float_to_rad(action.angle)

        x0, y0 = agent.position.to_tuple()
        new_position = Position(
            x = x0 + (agent.speed*action.x_intensity),
            y = y0 + (agent.speed*action.y_intensity)
        )
        agent.position = new_position

        state = self.get_state()
        return state, self.reward(state), agent.goal_reached()




env = gymnasium.make('SimulationEnv-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

TARGET_UPDATE = 10
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.shape[0]
state, info = env.reset()
n_observations = len(state)*2

policy_net = Network(n_observations, n_actions).to(device)
target_net = Network(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state)
    else:
        return torch.tensor(env.action_space.sample(), device=device, dtype=torch.float32)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float32)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Optimiser le modèle
    optimizer.zero_grad()

    for i in range(BATCH_SIZE):
        action = torch.tensor(batch.action[i], dtype=torch.float32, device=device)
        state = torch.tensor(batch.state[i], dtype=torch.float32, device=device)
        next_state = torch.tensor(batch.next_state[i], dtype=torch.float32, device=device)
        reward = torch.tensor(batch.reward[i], device=device)

        # Calculer les valeurs Q pour l'état actuel et l'action correspondante
        state_action_value = policy_net(state)

        # Calculer la récompense attendue pour l'échantillon actuel
        expected_state_action_value = (target_net(next_state).max(0).values * GAMMA) + reward

        # Calculer la perte entre la valeur prédite et la valeur cible
        loss = F.smooth_l1_loss(state_action_value, expected_state_action_value.unsqueeze(0))

        # Propager l'erreur et effectuer une étape d'optimisation
        loss.backward()

    optimizer.step()

    # Mettre à jour le réseau cible (target network)
    if steps_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())



if torch.cuda.is_available():
    num_episodes = 3
else:
    num_episodes = 2

for i_episode in range(num_episodes):
    print(f"Episode : {i_episode}")
    # Initialize the environment and get its state
    state, info = env.reset()

    agent_position = torch.tensor(state['agent'], dtype=torch.float32, device=device)
    target_position = torch.tensor(state['target'], dtype=torch.float32, device=device)
    state_tensor = torch.cat((agent_position, target_position), dim=0)
    
    state = torch.tensor(state_tensor, dtype=torch.float32, device=device)
    for t in count():
        action = select_action(state).numpy()
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        agent_position = torch.tensor(observation['agent'], dtype=torch.float32, device=device)
        target_position = torch.tensor(observation['target'], dtype=torch.float32, device=device)
        observation_tensor = torch.cat((agent_position, target_position), dim=0)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation_tensor, dtype=torch.float32, device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
