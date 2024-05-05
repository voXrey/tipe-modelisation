import numpy as np
import toml
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from enum import Enum
from utils import *

config_path = "test\config.toml"
config = toml.load(config_path)

match config:
    case {
        "exploration_rate": float()
    }:
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
    def __init__(self, position:Position, radius:float) -> None:
        self.position = position
        self.radius = radius

class Goal(Entity):
    def __init__(self, position:Position, radius:float) -> None:
        super().__init__(position, radius)

class Agent(Entity):
    def __init__(self, id:int, position:Position, radius:float, goal:Goal, state_dim, action_dim, save_dir) -> None:
        super().__init__(position, radius)
        self.id = id
        self.goal = goal
        self.simulation:Simulation = None
        
        self.speed = 1.0

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = AgentNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = config.get("exploration_rate")
        self.exploration_rate_decay = config.get("exploration_rate_decay")
        self.exploration_rate_min = config.get("exploration_rate_min")
        self.curr_step = 0

        self.save_every = 5e5

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gama = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def collision(self, entity:Entity) -> bool:
        return self.position.distance(entity.position) < max(self.radius, entity.radius)

    def goal_reached(self) -> bool:
        return self.collision(self.goal)

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = Action(np.random.uniform(0, 1), np.random.uniform(0, 1))
        
        # EXPLOIT
        else:
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = Action(action_values[0][0].item(), action_values[0][1].item())

        # decrease exploiration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action.angle, action.intensity])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"AgentNet saved to {save_path} at step {self.curr_step}")

class AgentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, _, _ = input_dim

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        input = input.float()
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
            
    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

class Simulation:

    def __init__(self, agents:list[Agent]=[]) -> None:
        self.agents:dict[int,Agent] = dict()
        for agent in agents: self.add_agent(agent)

        self.action_space = 2
    
    def get_action_space(self) -> tuple:
        self.action_space = len(self.agents)*2

    def add_agent(self, agent:Agent) -> None:
        agent.simulation = self
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id:int) -> None:
        del self.agents[agent_id]

    def get_agent(self, agent_id:int) -> None|Agent:
        try: return self.agents[agent_id]
        except: return

    def get_state(self):
        goal = self.agents[0].goal
        positions = [goal.position.x, goal.position.y]
        for a in self.agents.values():
            positions.append(a.position.x)
            positions.append(a.position.y)
        return torch.tensor(positions)

    def reset(self):
        for k in self.agents:
            self.agents[k].position = Position(500, 700)
        return self.get_state()

    def start(self):
        pass

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
    

