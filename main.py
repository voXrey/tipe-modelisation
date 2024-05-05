import torch
from new_model import *
from pathlib import Path
import datetime

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

env = Simulation()
agent = Agent(
    id = 0,
    position=Position(500, 700),
    radius=10,
    goal=Goal(Position(500, 500), 10),
    state_dim=(1, 1, 4),
    action_dim=env.action_space,
    save_dir=save_dir)
env.add_agent(agent)


episodes = 100
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done = env.step(agent, action)
        
        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        agent.learn()

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break


    if (e % 5 == 0) or (e == episodes - 1):
        print(f"EPSIODE :  {e}")
