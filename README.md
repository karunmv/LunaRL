# Reinforcement Learning for Lunabotics

This project aims to create a path planning model using q-learning for the Lunabotics 2025 competition. Started as a project idea for ECE533 (Advanced Robotics) at UMaine.

## About Code

- terrain_nav_env.py: Sets up the environment for the rover to navigate
- q_learning_train.py: Trains the agent to navigate the terrain using q-learning with the epsilon-greedy approach
- realistic_terrain_nav.py: Make the craters more realistic using the noise package (Still experimenting, not quite ready)
- gym_test.py: Just playing with the gymnasium package to test out its features to use in the actual project