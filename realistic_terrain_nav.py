import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2  # Install using `pip install noise`

class TerrainNavEnv(gym.Env):
    def __init__(self):
        super(TerrainNavEnv, self).__init__()

        self.grid_size = 30
        self.start_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, 0)

        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        self.terrain = np.zeros((self.grid_size, self.grid_size))

        # Add crater-style obstacles
        for _ in range(10):
            cx = np.random.randint(self.grid_size)
            cy = np.random.randint(self.grid_size)
            radius = np.random.randint(2, 5)
            self.add_crater(cx, cy, radius)

        # Add bumpy/rough terrain with Perlin noise
        self.generate_rough_terrain(scale=10.0, threshold=0.3)

        self.agent_pos = list(self.start_pos)
        self.agent_trail = [self.start_pos]

    def add_crater(self, center_x, center_y, radius):
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                        self.terrain[y, x] = 2  # Obstacle

    def generate_rough_terrain(self, scale=10.0, threshold=0.3):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                noise_val = pnoise2(x / scale, y / scale)
                if abs(noise_val) > threshold and self.terrain[y, x] == 0:
                    self.terrain[y, x] = 1  # Rough terrain

    def reset(self, seed=None, options=None):
        self.agent_pos = list(self.start_pos)
        self.agent_trail = [self.start_pos]
        return np.array(self.agent_pos), {}

    def step(self, action):
        x, y = self.agent_pos

        if action == 0 and y < self.grid_size - 1:  # Up
            y += 1
        elif action == 1 and y > 0:  # Down
            y -= 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # Right
            x += 1

        self.agent_pos = [x, y]
        self.agent_trail.append(self.agent_pos)

        reward = -1
        if self.terrain[y, x] == 1:
            reward = -3
        if self.terrain[y, x] == 2:
            reward = -100
        if (x, y) == self.goal_pos:
            reward = 20

        done = (x, y) == self.goal_pos
        return np.array(self.agent_pos), reward, done, False, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal_pos[1], self.goal_pos[0]] = 0.5
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.terrain[y, x] == 1:
                    grid[y, x] = 0.9
                elif self.terrain[y, x] == 2:
                    grid[y, x] = -0.1

        for (x, y) in self.agent_trail:
            grid[y, x] = 1.0

        plt.imshow(grid, cmap="tab20c", origin="lower")
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.grid(True)
        trail_x, trail_y = zip(*self.agent_trail)
        plt.scatter(trail_x, trail_y, color='r', s=50, marker='o', label='Agent Trail')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()

    def render_q_path(self, q_table):
        pos = list(self.start_pos)
        path = [tuple(pos)]
        visited = set()
        steps = 0
        max_steps = self.grid_size * self.grid_size

        while tuple(pos) != self.goal_pos and steps < max_steps:
            x, y = pos
            if (x, y) in visited:
                break
            visited.add((x, y))

            action = np.argmax(q_table[x, y])
            if action == 0 and y < self.grid_size - 1:
                y += 1
            elif action == 1 and y > 0:
                y -= 1
            elif action == 2 and x > 0:
                x -= 1
            elif action == 3 and x < self.grid_size - 1:
                x += 1
            pos = [x, y]
            path.append((x, y))
            steps += 1

        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal_pos[1], self.goal_pos[0]] = 0.5
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.terrain[y, x] == 1:
                    grid[y, x] = 0.9
                elif self.terrain[y, x] == 2:
                    grid[y, x] = -0.1
        for (x, y) in self.agent_trail:
            grid[y, x] = 1.0

        plt.imshow(grid, cmap="tab20c", origin="lower")
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.grid(True)

        trail_x, trail_y = zip(*self.agent_trail)
        plt.scatter(trail_x, trail_y, color='r', s=50, marker='o', label='Agent Trail')
        path_x, path_y = zip(*path)
        plt.scatter(path_x, path_y, color='blue', s=50, marker='x', label='Q Greedy Path')

        plt.legend(loc="upper right")
        plt.show(block=False)
        plt.pause(0.5)
        plt.clf()
