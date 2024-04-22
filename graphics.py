from classes import Position, Model, Agent, Step


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation



def visualize_steps(steps: list[Step]) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 110])
    ax.set_ylim([-10, 110])
    ax.set_aspect("equal")
    agents = {agent.id: agent for step in steps for agent in step.agents}
    circles = {agent.id: patches.Circle(agents[agent.id].position.to_tuple(), agent.rayon, fill=False) for agent in agents.values()}
    for circle in circles.values():
        ax.add_patch(circle)

    def update(frame):
        for agent in steps[frame].agents:
            circles[agent.id].center = agents[agent.id].position.to_tuple()
        return circles.values()

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000, blit=True)
    plt.show()

# Exemple
agents = [
    Agent(0, Position(34, 14)),
    Agent(1, Position(89, 53))
]

agents2 = [
    Agent(0, Position(0, 0)),
    Agent(1, Position(100, 100))
]
s = [
    Step(0, agents),
    Step(1, agents2),
    
]

visualize_steps(s)