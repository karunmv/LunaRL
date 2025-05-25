import numpy as np
from realistic_terrain_nav import TerrainNavEnv

# Hyperparameters
episodes = 50000
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.05

# Setup environment
env = TerrainNavEnv()

# Initialize Q-table
q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        x, y = state
        return np.argmax(q_table[x, y])

# Training loop
for ep in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        x, y = state
        nx, ny = next_state
        
        # Q-learning update
        q_table[x, y, action] = q_table[x, y, action] + alpha * (reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action])
        
        state = next_state
    
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Render every 50 episodes
    if ep % 50 == 0:
        print(f"Episode {ep} complete. Epsilon: {epsilon:.3f}")
        env.render()
        #env.render_q_path()

env.close()
print("Training complete!")
