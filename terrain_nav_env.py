import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class TerrainNavEnv(gym.Env):
    def __init__(self):
        super(TerrainNavEnv, self).__init__()
        
        self.grid_size = 30 #Define grid size
        self.start_pos = (0, 0)
        self.goal_pos = (self.grid_size-1, 0)
        
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # Terrain map (0 = normal, 1 = rough, 2 = obstacle)
        self.terrain = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size // 2 + 1):
            self.terrain[i, self.grid_size // 2] = 2


        for j in range(self.grid_size):
            self.terrain[np.random.random_integers(self.grid_size-1), np.random.random_integers(self.grid_size-1)] = 2  # random obstacles
        
        for k in range(self.grid_size):
            a = np.random.random_integers(self.grid_size-1)
            b = np.random.random_integers(self.grid_size-1)
            if self.terrain[a, b] == 0:
                self.terrain[a, b] = 1 # random rough patches
        
        self.agent_pos = list(self.start_pos)
        self.agent_trail = [self.start_pos]  # Keeps track of rover's path

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
        self.agent_trail.append(self.agent_pos)  # Add new position to the trail
        
        # Calculate reward
        reward = -1  # Default move cost
        if self.terrain[y, x] == 1:
            reward = -3  # Rough terrain penalty
        if self.terrain[y, x] == 2:
            reward = -100  # Obstacle penalty
        if (x, y) == self.goal_pos:
            reward = 20  # Reached goal
        
        done = (x, y) == self.goal_pos
        
        return np.array(self.agent_pos), reward, done, False, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.goal_pos[1], self.goal_pos[0]] = 0.5  # Goal
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.terrain[y,x] == 1:
                    grid[y,x] = 0.9  # Rough terrain
                if self.terrain[y,x] == 2:
                    grid[y,x] = -0.1  # Obstacle

        # Plot the agent's path (trail) as red dots
        for (x, y) in self.agent_trail:
            grid[y, x] = 1.0  # Rover trail in red
        
        # Make the plot for the grid
        plt.imshow(grid, cmap="tab20c", origin="lower")
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))
        plt.grid(True)
        
        # Display trail as dots on the grid
        trail_x, trail_y = zip(*self.agent_trail)
        plt.scatter(trail_x, trail_y, color='r', s=50, marker='o')  # Red dots for trail
        
        # Display the plot (without blocking further code execution)
        plt.show(block=False)
        plt.pause(0.1)  # Pause to allow for animation
        plt.clf()  # Clear the plot for the next frame