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

config_path = Path("config.toml")
config = toml.load(config_path)

match config:
    case {}:
        pass
    case _:
        raise ValueError(f"invalid configuration: {config}")


class Action:
    def __init__(self, angle:float, intensity:float):
        self.angle = angle
        self.intensity = intensity

class Position:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def distance(self, position):
        return np.sqrt((self.x-position.x)**2 + (self.y-position.y)**2)

    def __add__(self, o):
        self.x += o.x
        self.y += o.y
        return 
    
    def __str__(self) -> str:
        return f"{self.x}:{self.y}"
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y)

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
            x = x0 + np.cos(angle)*(agent.speed*action.intensity),
            y = y0 + np.sin(angle)*(agent.speed*action.intensity)
        )
        agent.position = new_position

        state = self.get_state()
        return state, self.reward(state), agent.goal_reached()


class Main:
    def __init__(self):
        self.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LinearRegressionModel(1)

class LinearRegressionModel(nn.Module):
    def __init__(self, agents_number:int):
        super(LinearRegressionModel, self).__init__()
        
        self.agents_number = agents_number
        self.layer1 = nn.Linear(2*(agents_number+1), 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 2*agents_number)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

model = LinearRegressionModel()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    