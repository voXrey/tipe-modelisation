import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from classes import Position


SPEED = 5.0
RADIUS = 10.0

class SimulationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size:Position=Position(512, 512)):
        self.max_pos = max(size.x, size.y)
        self.size = size  # The bottom-rigth corner of the grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.max_pos, shape=(2,), dtype=np.float32),
                "target": spaces.Box(0, self.max_pos, shape=(2,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.uniform(0, self.max_pos, size=(2,))

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while self._get_info()["distance"] <= 30:
            self._target_location = self.np_random.uniform(0, self.max_pos, size=(2,))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + action*SPEED, 0, self.max_pos - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = self._get_info()["distance"] <= RADIUS
        reward = 1 if terminated else (1 - self._get_info()["distance"]/self.max_pos)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size.x, self.size.y)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.max_pos
        )

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self._target_location,
            RADIUS
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location,
            RADIUS,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

register(
    id="SimulationEnv-v0",
    entry_point="simulation_env:SimulationEnv",
    max_episode_steps=300,
)