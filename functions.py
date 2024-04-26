from classes import *


barem = {
    "collide": 2,
    "distance": 1,
    "time": 10
}


def score_step(previous_step:Step, step:Step, goal:Position):
    s = 0
    for agent in previous_step.agents:
        s += (agent.count_collisions(previous_step.agents)-1)*barem["collide"]
        s += (agent.position.distance(goal)) * barem["distance"]

def score(steps:list[Step]):
    s = 0
    for step in steps:
        s += score_step(step)
    s += len(steps) * barem["time"]

def generate(first_step:Step):
    