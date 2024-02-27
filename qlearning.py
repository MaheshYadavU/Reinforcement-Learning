import numpy as np
import gym
import time
import math
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="rgb_array")
print("Action_Spaces", env.action_space.n)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4000
total = 0
total_reward = 0
prior_reward = 0
Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
epsilon = 1
print("Shape of np_array_win_size:", np_array_win_size.shape)
epsilon_decay_value = 0.99995
q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape
episode_rewards = []

def get_discrete_state(state):
    if isinstance(state, tuple):
        state_array = state[0]
    else:
        state_array = state
    state_padded = np.pad(state_array, (0, 4 - len(state_array)), mode='constant')
    discrete_state = state_padded / np_array_win_size + np.array([15, 10, 1, 10])
    discrete_state = discrete_state[:,].astype(bool)
    return discrete_state

for episode in range(EPISODES + 1):
    t0 = time.time()
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    if episode % 4000 == 0:
        print("Episode: " + str(episode))
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        step_action = env.step(action)
        new_state, reward, done, _ = step_action[:4]
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if episode % 4000 == 0:
            env.render()
        if not done:
            discrete_state = new_discrete_state
            new_discrete_state = new_discrete_state.astype(int)
            new_discrete_state = np.clip(new_discrete_state, 0, q_table.shape[0] - 1)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

    if epsilon > 0.05:
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)
            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

    t1 = time.time()
    episode_total = t1 - t0
    total = total + episode_total

    total_reward += episode_reward
    episode_rewards.append(episode_reward)
    prior_reward = episode_reward

    if episode % 1000 == 0:
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

plt.plot(episode_rewards)
plt.title('Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
env.close()
