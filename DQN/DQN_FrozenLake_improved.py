import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)        # deque : double-ended que
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))   # deque.append()는 list.append()와 동일하게 작동한다. 
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) # Replay buffer에서 random sampling 이후 unpacking해서 각 요소들끼리 모은다. 
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) # doen즉 종료 여부는 빼도 되지 싶다. => 어짜피 한 에피소드 기준으로 학습을 시킬 것이다.

# Network 정의 (MLP로 변경)
class Network(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=64):
        super(Network, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = torch.Tensor(x) if not torch.is_tensor(x) else x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.one_hot(x.to(torch.int64), num_classes=self.n_states).to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ENV 및 Parameter Setting
env = gymnasium.make('FrozenLake-v1', is_slippery=False)  # 미끄러짐 비활성화
n_states = env.observation_space.n
n_actions = env.action_space.n
num_episode = 1000  # 에피소드 수 증가
max_step = 100
epsilon_start = 1.0  # 초기 epsilon
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start
gamma = 0.99
lr = 0.001  # 학습률 감소
batch_size = 32  # 배치 크기 증가
buffer_capacity = 10000  # 버퍼 크기 증가
target_update_freq = 10  # 타겟 네트워크 업데이트 주기

# Model, Loss, Optimizer, Replay Buffer 생성
q_network = Network(n_states=n_states, n_actions=n_actions)
target_network = Network(n_states=n_states, n_actions=n_actions)
target_network.load_state_dict(q_network.state_dict())
criterion = nn.MSELoss()  # MSE만 사용
optimizer = SGD(params=q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# Training Loop
for i in range(num_episode):
    current_state, _ = env.reset()
    all_reward = 0
    done = False
    step = 0

    while not done and step < max_step:
        # Epsilon-Greedy
        if torch.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            q_network.eval()
            with torch.no_grad():
                Qs = q_network(current_state)
                action = torch.argmax(Qs).item()
            q_network.train()

        # Environment step
        next_state, reward, done, _, _ = env.step(action)
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

            # Q_value 계산
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

            print(f"Episode: {i + 1}, Step: {step}, Loss: {loss.item():.4f}, Reward: {reward}, Done: {done}")

        current_state = next_state
        step += 1

        # Target Network 업데이트
        if step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

    # Epsilon Decay
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    print(f"Episode {i + 1}, Total Reward: {all_reward}, Epsilon: {epsilon:.4f}")

env.close()