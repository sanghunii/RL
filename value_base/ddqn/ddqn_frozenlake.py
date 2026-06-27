import gymnasium
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from collections import deque
import random


    
# Define network 
class Network(nn.Module):
    """
    DQN network for frozen-lake
    """

    def __init__(self, n_states, n_actions): 
        """
        n_states: 환경의 총 상태 수 (원-핫 인코딩의 num_classes 및 입력 차원으로 활용)
        n_actions: 가능한 행동의 수 (최종 Q-value 출력 차원)
        """
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        
        # === Output Head (Q-value Head) ===
        # 원-핫 인코딩된 상태(n_states 차원)를 받아 각 Action에 대한 Q-value를 계산합니다.
        # 기존 코드의 nn.Linear(n_states, n_actions)와 동일하게 동작하도록 nn.Sequential로 캡슐화했습니다.
        self.q_head = nn.Sequential(
            nn.Linear(in_features=self.n_states, out_features=self.n_actions)
        )


    def forward(self, state):
        state = torch.tensor(state) if not torch.is_tensor(state) else state
        
        if state.dim() == 0:
            state = state.unsqueeze(0)
            
        # 3. 원-핫 인코딩 및 타입 변환
        features = F.one_hot(state.to(torch.int64), num_classes=self.n_states).to(torch.float32)
        
        # 4. Q-value 계산
        q_values = self.q_head(features)
        
        return q_values


    def get_q_values(self, state):
        """
        Aggregating Module
        - 외부에서 일관된 메서드명으로 Q값을 호출할 수 있도록 캡슐화.
        """
        return self.forward(state)



# Replay Buffer 정의 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) # Replay buffer에서 random sampling 이후 unpacking해서 각 요소들끼리 모은다. 
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) # doen즉 종료 여부는 빼도 되지 싶다. => 어짜피 한 에피소드 기준으로 학습을 시킬 것이다.
    



# ENV 및 Parameter 셋팅
env = gymnasium.make('FrozenLake-v1')
n_states = env.observation_space.n
n_actions = env.action_space.n
num_episode = 1
max_step = 20  
exploration_prob = 0.5 
gamma = 0.99       
lr = 0.1        
batch_size = 5  
buffer_capacity = 1000 
delta = 1.0


# Model, Loss, Optimizer, Replay Buffer 생성
q_network = Network(n_states=n_states, n_actions=n_actions)
target_network = Network(n_states=n_states, n_actions=n_actions)
target_network.load_state_dict(q_network.state_dict())
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss() 
optimizer = SGD(params=q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=buffer_capacity)



import torch.nn.functional as F

# Training Loop
for i in range(num_episode):
    current_state, _ = env.reset()
    all_reward = 0
    done = False 
    step = 0

    # 1. & 연산자 대신 and 사용
    while not done and step < max_step:
        q_network.eval()

        # Epsilon-greedy 정책
        if torch.rand(1) < exploration_prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                # 3. 리팩토링한 메서드 활용
                Qs = q_network.get_q_values(current_state) 
                action = torch.argmax(Qs).item()
        
        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action=action)
        done = terminated or truncated
        all_reward += reward

        # Replay buffer에 저장
        replay_buffer.push(current_state, action, reward, next_state, done)

        # Replay buffer에서 샘플링 및 학습
        if len(replay_buffer) >= batch_size:
            q_network.train()
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=batch_size)

            # 4. 데이터 타입을 파이토치 기본 연산 타입인 float32로 통일
            states = torch.tensor(states, dtype=torch.float32) 
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32) # (1 - dones) 연산을 위해 float32 사용
            
            q_values = q_network(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q_values = target_network(next_states)
                max_next_q = next_q_values.max(1)[0]
                # 5. [치명적 버그 수정] reward(단일)가 아닌 rewards(배치) 사용
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # Loss 계산 및 업데이트
            optimizer.zero_grad()
            
            # 6. 주석 내용에 맞춰 실제 Huber Loss(Smooth L1 Loss) 적용
            loss = F.smooth_l1_loss(q_value, target_q) 
            
            loss.backward()
            optimizer.step()
            
        current_state = next_state
        step += 1

    # 에피소드 종료 후 타겟 네트워크 업데이트 
    target_network.load_state_dict(q_network.state_dict())
    
    # 디버깅/출력용 (너무 자주 출력되면 학습 속도가 느려질 수 있으니 주기적 출력을 권장합니다)
    print(f"Episode {i + 1}, Total Reward: {all_reward}")

env.close()