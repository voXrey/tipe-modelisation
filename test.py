import numpy as np


class Position:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def distance(self, position):
        return np.sqrt((self.x-position.x)**2 + (self.y-position.y)**2)

    def __add__(self, o):
        self.x += o.x
        self.y += o.y
        return self

class Agent:
    speed = 1.0
    rayon = 5

    def __init__(self, id:int, position:Position) -> None:
        self.id = id
        self.position = position
        self.model:Model = None
        self.dp = Position(0,0) # Décalage avec la prochaine position

    def future_collision(self, agent) -> bool:
        

class Model:
    agents:dict[int,Agent]

    def __init__(self, goal_position:Position, agents:list[Agent]=[]) -> None:
        self.goal_position = goal_position
        for agent in agents: self.add_agent(agent)
    
    def add_agent(self, agent:Agent) -> None:
        agent.model = self
        self.agents[agent.id] = agent
    
    def next_pos_agents(self):
        to_move = np.random.shuffle(self.agents.keys())
        moved = dict()

        for agent in self.agents.values():





