# PettingZoo API 说明文档

PettingZoo 是一个多智能体强化学习环境库，采用 **AEC (Agent Environment Cycle)** 架构，支持各种多智能体 RL 场景。

## 核心属性

### agents
当前所有活跃代理的名称列表，通常为整数。代理可能随着环境推进而变化（可添加或移除）。

```python
agents = env.agents  # ['agent_0', 'agent_1', ...]
```

### num_agents
代理列表的长度。

```python
num = env.num_agents
```

### agent_selection
当前被选中的代理名称，当前可以对它执行动作。

```python
current_agent = env.agent_selection
```

### observation_space(agent)
获取指定代理的观察空间。该空间对特定代理 ID 永远不变。

```python
obs_space = env.observation_space(agent)
```

### action_space(agent)
获取指定代理的动作空间。该空间对特定代理 ID 永远不变。

```python
act_space = env.action_space(agent)
```

## 状态属性

### terminations
一个字典，包含当前所有代理的终止状态，键为代理名称。

```python
terminations = env.terminations
# {'agent_0': False, 'agent_1': True, ...}
```

### truncations
一个字典，包含当前所有代理的截断状态（通常表示环境达到最大步数），键为代理名称。

```python
truncations = env.truncations
# {'agent_0': False, 'agent_1': False, ...}
```

### rewards
一个字典，包含每个代理的即时奖励，键为代理名称。

```python
rewards = env.rewards
# {'agent_0': 1.0, 'agent_1': -0.5, ...}
```

### infos
一个字典，包含每个代理的附加信息，每个代理的信息本身也是一个字典。

```python
infos = env.infos
# {'agent_0': {'some_info': ...}, 'agent_1': {...}, ...}
```

## 核心方法

### observe(agent)
返回代理当前可以获得的观察。

```python
observation = env.observe(agent)
```

### seed(seed=None)
重置环境的随机种子。`reset()` 必须在 `seed()` 之后、`step()` 之前调用。

```python
env.seed(42)
env.reset()
```

### render()
返回环境的渲染帧。
- `render_mode="rgb_array"`: 返回 numpy 数组
- `render_mode="ansi"`: 返回字符串
- `render_mode="human"`: 人类可读模式（无需调用）

```python
frame = env.render()  # rgb_array 模式
env.render()         # human 模式
```

### close()
关闭渲染窗口。

```python
env.close()
```

## 高级 API（推荐使用）

### last()
获取当前选中代理的最后观察、奖励、终止/截断状态和信息。`agent_iter()` 循环中常用。

```python
observation, reward, termination, truncation, info = env.last()
```

### agent_iter(max_iter=...)
迭代所有代理。返回生成器，每次迭代 yield 当前代理名称。

```python
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
```

### step(action)
执行动作。环境会推进到下一个状态。

```python
env.step(action)
```

## 使用示例

```python
from pettingzoo.butterfly import cooperative_pong_v5
import time

# 初始化环境
env = cooperative_pong_v5.env(render_mode="human")

# 重置环境（可指定种子）
env.reset(seed=42)

# 代理迭代循环
for agent in env.agent_iter():
    # 获取当前代理的最后观察
    observation, reward, termination, truncation, info = env.last()

    # 如果代理已终止或截断，动作设为 None
    if termination or truncation:
        action = None
    else:
        # 从动作空间采样
        action = env.action_space(agent).sample()

    # 执行动作
    env.step(action)

    # 渲染（human 模式不需要手动调用）
    env.render()
    time.sleep(0.02)  # 给渲染窗口刷新时间

# 关闭环境
env.close()
```

## 注意事项

1. **AEC 架构**: PettingZoo 采用 AEC 模式，代理按顺序执行，而非同时执行
2. **动作设置**: 当 `termination` 或 `truncation` 为 `True` 时，`action` 应设为 `None`
3. **种子设置**: `seed()` 后必须调用 `reset()`，且在 `step()` 之前
4. **渲染模式**: `human` 模式下不需要手动调用 `render()`
5. **清理资源**: 使用完环境后务必调用 `close()`
