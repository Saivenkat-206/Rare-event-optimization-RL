import numpy as np
from env import GridEnv

def run_policy(env, policy_fn, episodes=50):
    total_rewards = []
    total_failures = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        failures = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            if info["failure"]:
                failures += 1

        total_rewards.append(ep_reward)
        total_failures.append(failures)

    return (np.mean(total_rewards), np.mean(total_failures), np.min(total_rewards))

env = GridEnv()

# Policies
always_maintain = lambda obs: 1
never_maintain = lambda obs: 0
random_policy = lambda obs: env.action_space.sample()

for name, policy in [
    ("Always Maintain", always_maintain),
    ("Never Maintain", never_maintain),
    ("Random", random_policy)
]:
    avg_reward, avg_failures, worst_case = run_policy(env, policy)
    print(f"{name}:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Failures: {avg_failures:.2f}")
    print(f"  Worst-case Reward: {worst_case:.2f}")
    print("-" * 40)
    