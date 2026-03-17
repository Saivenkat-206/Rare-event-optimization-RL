import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("episode_rewards.npy")

print("Mean:", np.mean(rewards))
print("Std:", np.std(rewards))
print("Worst:", np.min(rewards))
print("Best:", np.max(rewards))

plt.figure()
plt.hist(rewards, bins=25)
plt.xlabel("Episode Return")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Returns (Naïve DQN)")
plt.show()