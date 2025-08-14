import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from collections import deque
import random

# Network 정의 (기존과 동일)
class Network(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.linear = nn.Linear(in_features=n_states, out_features=self.n_actions)

    def forward(self, x):
        x = torch.Tensor(x) if not torch.is_tensor(x) else x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.one_hot(x.to(torch.int64), num_classes=self.n_states).to(torch.float32)
        y = self.linear(x)
        return y



# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

# ENV 및 파라미터 셋팅
env = gymnasium.make("FrozenLake-v1")
n_states = env.observation_space.n
n_actions = env.action_space.n
num_episodes = 1
max_steps = 20
exploration_prob = 0.5
gamma = 0.99
lr = 0.1
batch_size = 5
buffer_capacity = 1000

# 모델, Loss, Optimizer, Replay Buffer 생성
q_network = Network(n_states=n_states, n_actions=n_actions)
target_network = Network(n_states=n_states, n_actions=n_actions)
target_network.load_state_dict(q_network.state_dict())  # 초기 가중치 동기화
criterion = nn.MSELoss()
# + MAE 
optimizer = SGD(params=q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=buffer_capacity)




# Training Loop
for i in range(num_episodes):
    current_state, _ = env.reset()
    all_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        # Action 선택 (epsilon-greedy)
        if torch.rand(1) < exploration_prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                Qs = q_network(current_state)
                action = torch.argmax(Qs).item()

        # Environment step
        next_state, reward, done, _, _ = env.step(action=action)
        all_reward += reward

        # Replay buffer에 저장
        replay_buffer.push(current_state, action, reward, next_state, done)

        # Replay buffer에서 샘플링 및 학습
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Q값 계산
            q_values = q_network(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q값 계산
            with torch.no_grad():
                next_q_values = target_network(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)

            # Loss 계산 및 업데이트
            optimizer.zero_grad()
            loss = criterion(q_value, target_q)
            loss.backward()
            optimizer.step()

            print(f"Step: {step}, Loss: {loss.item():.4f}, Reward: {reward}, Done: {done}")

        current_state = next_state
        step += 1

    # 에피소드 종료 후 타겟 네트워크 업데이트
    target_network.load_state_dict(q_network.state_dict())
    print(f"Episode {i+1}, Total Reward: {all_reward}")

env.close()