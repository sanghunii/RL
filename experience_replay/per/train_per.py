import gymnasium
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import random
from typing import Tuple, List
from value_base.per.per_buffer import PrioritizedReplayBuffer

class Network(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.linear = nn.Linear(in_features=n_states, out_features=self.n_actions)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.int64) if not torch.is_tensor(x) else x.to(torch.int64)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.one_hot(x, num_classes=self.n_states).to(torch.float32)
        y = self.linear(x)
        return y


# ==========================================
# ENV 및 하이퍼파라미터 세팅
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gymnasium.make('FrozenLake-v1')
n_states = env.observation_space.n  
n_actions = env.action_space.n 

num_episode = 200    # PER 효과를 보기 위해 에피소드 수 상향
max_step = 50        
exploration_prob = 0.3  
gamma = 0.99        
lr = 0.05        
batch_size = 16      
buffer_capacity = 2000  

# PER 하이퍼파라미터 및 선형 어닐링(Linear Annealing) 설정
alpha = 0.6
beta_start = 0.4
beta_end = 1.0
total_steps = num_episode * max_step  # 대략적인 총 글로벌 스텝 수
global_step = 0

# 객체 생성
q_network = Network(n_states=n_states, n_actions=n_actions).to(device)
target_network = Network(n_states=n_states, n_actions=n_actions).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = SGD(params=q_network.parameters(), lr=lr)

# 오리지널 ReplayBuffer 대신 우리가 만든 PrioritizedReplayBuffer 선언!
replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha)


# ==========================================
# Training Loop
# ==========================================
for i in range(num_episode):
    current_state, _ = env.reset()      
    all_reward = 0
    done = False 
    step = 0

    while not done and step < max_step:
        global_step += 1
        
        # 1. Action 선택 
        q_network.eval()
        if random.random() < exploration_prob:
            action = env.action_space.sample()  
        else:
            with torch.no_grad():
                Qs = q_network(torch.tensor([current_state], device=device))
                action = torch.argmax(Qs).item()
        
        # 2. Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        all_reward += reward

        # 3. PER 버퍼에 저장
        replay_buffer.push(current_state, action, reward, next_state, done)

        # 4. 샘플링 및 학습 프로세스
        if replay_buffer.tree.n_entries >= batch_size:
            q_network.train()
            
            # Beta값 Linear Annealing 계산
            beta = min(beta_end, beta_start + (beta_end - beta_start) * (global_step / total_steps))

            # PER 버퍼 샘플링 (is_weights와 tree_idx를 함께 반환받음)
            states, actions, rewards, next_states, dones, is_weights, tree_idx = replay_buffer.sample(
                batch_size=batch_size, beta=beta, device=device
            )
            
            # Predict Q-value 계산
            q_values = q_network(states)    
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) 
            
            # Target Q-value 계산
            with torch.no_grad():
                next_q_values = target_network(next_states)     
                max_next_q = next_q_values.max(1)[0]            
                target_q = rewards + gamma * max_next_q * (~dones)
            
            # 5. TD-Error 및 중요도 샘플링이 반영된 Loss 계산 (핵심 킬러 파트!)
            # 개별 데이터별 요소를 살리기 위해 reduction='none'으로 오차 계산
            td_errors = torch.abs(target_q - q_value)
            
            # 칼럼 내용 반영: 수식 (5), (6)에 의거하여 w * Loss 형태 구현 (여기서는 Huber Loss 대용으로 MSE/L1 분기 처리)
            # 예측값 배열 전체를 다 비교하는 대신, 업데이트 대상인 Q(s,a)와 target_q 간의 오차 기반으로 분기
            if torch.mean(td_errors).item() < 1.0:
                raw_loss = F.mse_loss(q_value, target_q, reduction='none')
            else:
                raw_loss = F.l1_loss(q_value, target_q, reduction='none')
                
            # 최종 가중치 반영 후 평균 전개
            loss = torch.mean(is_weights * raw_loss)
            
            # 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 6. 중요! 학습 후 TD-Error를 이용해 SumTree의 우선순위 최신화
            # 연산 효율을 위해 CPU numpy 배열로 변환하여 전달
            abs_errors_numpy = td_errors.detach().cpu().numpy()
            replay_buffer.batch_update(tree_idx, abs_errors_numpy)

        current_state = next_state 
        step += 1

    # 에피소드 종료 후 타겟 네트워크 동기화
    target_network.load_state_dict(q_network.state_dict())

env.close()