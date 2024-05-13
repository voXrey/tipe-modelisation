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
        return 
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y)

class Action:
    def __init__(self, x_intensity:float, y_intensity:float):
        self.x_intensity = x_intensity
        self.y_intensity = y_intensity

class Entity:
    def __init__(self, position:Position=Position(0, 0), radius:float=0, speed:float=0) -> None:
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

class State:
    def __init__(self, goal:Goal, agents:list[Agent]):
        self.goal = goal
        self.agents = agents
    
    def numpy(self) -> np.ndarray:
        return np.array(
            []
        )

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
