from pettingzoo.butterfly import cooperative_pong_v5
import time

env = cooperative_pong_v5.env(render_mode="human")

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

    env.render()
    time.sleep(0.02)   # 关键：给pygame刷新时间

env.close()
