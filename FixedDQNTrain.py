import torch
import numpy as np
from env import GridEnv
from dqn import QNetwork, ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

env = GridEnv()
state_dim = 3
action_dim = 2

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.99
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
target_update_freq = 500
total_steps = 25000

state, _ = env.reset()

episode_rewards = []
current_reward = 0
episode = 0

for step in range(total_steps):

    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(q_net(s)).item()

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    buffer.push(state, action, reward, next_state, done)

    current_reward += reward
    state = next_state

    if done:
        episode_rewards.append(current_reward)
        episode += 1
        print(f"Episode {episode:3d} |"
              f"failure: {info['failure']} |"
              f"Reward: {current_reward:8.2f}")
        current_reward = 0
        state, _ = env.reset()

    if len(buffer) > batch_size:
        s, a, r, ns, d = buffer.sample(batch_size, device)

        current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = target_net(ns).max(1)[0]
            target_q = r + gamma * max_next_q * (1 - d)

        loss = torch.nn.functional.mse_loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step > 0 and step % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

print("\nTraining complete.")
print("Mean reward:", np.mean(episode_rewards))
print("Worst episode:", np.min(episode_rewards))

np.save("episode_rewards.npy", np.array(episode_rewards))
torch.save(q_net.state_dict(), "FixedDQN.pth")

