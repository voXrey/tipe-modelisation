import numpy as np
from copy import deepcopy


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

class Agent:

    def __init__(self, id:int, position:Position) -> None:
        self.id = id
        self.position = position
        self.model:Model = None

        self.speed = 1.0
        self.rayon = 5

        self.dp = Position(0,0) # Décalage avec la prochaine position

    def future_collision(self, agent) -> bool:
        return (self.position + self.dp).distance(agent.position + agent.dp) < self.rayon
    
    def collision(self, agent) -> bool:
        return (self.position + self.dp).distance(agent.position) < self.rayon

    def count_collisions(self, agents:list) -> int:
        i = 0
        for agent in agents:
            if self.collision(agent): i+=1
        return i

    def __str__(self) -> str:
        return f"{self.id},{self.position},{self.speed},{self.rayon}"


class Model:

    def __init__(self, goal_position:Position, agents:list[Agent]=[]) -> None:
        self.goal_position = goal_position
        self.agents:dict[int,Agent] = dict()
        for agent in agents: self.add_agent(agent)
    
    def add_agent(self, agent:Agent) -> None:
        agent.model = self
        self.agents[agent.id] = agent
    
    def next_pos_agents(self):
        to_move = self.agents.copy()
        agents_list:list[Agent] = list(self.agents.values())
        np.random.shuffle(agents_list)
        moved = dict()

        def there_is_collide(agent:Agent) -> bool:
            for a in moved.values():
                if agent.future_collision(a): return True
            for a in to_move.values():
                if agent.collision(a): return True
            return False
        
        for agent in agents_list:
            if agent.position.x > self.goal_position.x and agent.position.y >= self.goal_position.y:
                angle = np.pi+ np.arctan((agent.position.y-self.goal_position.y)/(agent.position.x-self.goal_position.x))
            elif agent.position.x < self.goal_position.x:
                angle = np.arctan((agent.position.y-self.goal_position.y)/(agent.position.x-self.goal_position.x))
            elif agent.position.x > self.goal_position.x:
                angle = np.pi/2-np.arctan((agent.position.y-self.goal_position.y)/(agent.position.x-self.goal_position.x))
            elif agent.position.y <= self.goal_position.y:
                angle = np.pi/2
            else:
                angle = -np.pi/2

            agent.dp = Position(np.cos(angle)*agent.speed, np.sin(angle)*agent.speed)
            print(angle)
            i = 0
            move = False
            dangle = 0
            while -np.pi/2 <= dangle <= np.pi/2:
                if there_is_collide(agent):
                    dangle -= (np.pi/8) * (i+1) * (-1)**(i)
                    agent.dp = Position(np.cos(angle+dangle)*agent.speed, np.sin(angle+dangle)*agent.speed)
                    i += 1
                    continue
                move = True
                break

            if not move: agent.dp = Position(0,0)

            moved[agent.id] = agent
            del to_move[agent.id]



class Step:
    def __init__(self, id:int, agents:list[Agent]):
        self.id = id
        self.agents = agents
    
    def __str__(self) -> str:
        l = list(map(str,self.agents))
        return f"{self.id};{';'.join(l)}"