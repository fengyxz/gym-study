import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")

# reset
observation, info = env.reset()

print(f"Starting ovservation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(action)
    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    """
    Reward 计算方式：

    observation, reward, terminated, truncated, info = env.step(action)

    - 每个 timestep：+1（只要杆子没倒下）
    - terminated = True：杆子倒下或小车位置/角度超出安全范围
    - truncated = True：达到时间限制（CartPole-v1 默认 500 步）
    """
    total_reward += reward
    episode_over = terminated or truncated
    
print(f"Episode finished! Total reward: {total_reward}")
env.close()
