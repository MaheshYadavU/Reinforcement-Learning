

This repository contains implementations of Q-Learning and DQN algorithms in Python using the OpenAI Gym environment. The code demonstrates how to train agents to solve the CartPole-v1 environment, a classic control problem, using both traditional Q-Learning and DQN approaches. Users can experiment with different hyperparameters, network architectures, and training strategies to understand and explore the capabilities of these algorithms in reinforcement learning tasks.

**Q-Learning**:
Q-Learning is a model-free reinforcement learning algorithm used to learn the optimal action-selection policy for a given Markov decision process (MDP). It learns a Q-value function that represents the expected cumulative reward for taking a particular action in a given state. The Q-value function is iteratively updated using the Bellman equation until convergence to approximate the optimal Q-values.

**Deep Q-Network (DQN)**:
Deep Q-Network (DQN) is an extension of Q-Learning that utilizes a neural network, known as the Q-network, to approximate the Q-value function. This enables the handling of high-dimensional state spaces and improves generalization. DQN employs techniques such as experience replay and target networks to stabilize training and improve sample efficiency. By combining Q-Learning with deep neural networks, DQN has achieved remarkable success in solving a wide range of reinforcement learning tasks, including playing Atari games and controlling robotic systems.

