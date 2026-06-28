import gymnasium
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from experience_replay.replay_buffer import ReplayBuffer



# 1. Define Network (변경 없음)
class Network(nn.Module):
    """
    DQN network for frozen-lake
    """
    def __init__(self, n_states, n_actions): 
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.q_head = nn.Sequential(
            nn.Linear(in_features=self.n_states, out_features=self.n_actions)
        )

    def forward(self, state):
        state = torch.tensor(state) if not torch.is_tensor(state) else state
        
        if state.dim() == 0:
            state = state.unsqueeze(0)
            
        features = F.one_hot(state.to(torch.int64), num_classes=self.n_states).to(torch.float32)
        q_values = self.q_head(features)
        
        return q_values

    def get_q_values(self, state):
        return self.forward(state)


# 2. ENV 및 Parameter 셋팅 (변경 없음)
env = gymnasium.make('FrozenLake-v1', is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

num_episode = 5000
max_step = 100  

# Epsilon Decay 로직을 위한 파라미터 셋팅
exploration_prob = 1.0         # 처음에는 100% 확률로 랜덤 탐험
min_exploration_prob = 0.01    # 아무리 학습이 끝나도 1%의 엉뚱한 행동(탐험)은 남겨둠
exploration_decay = 0.999      # 매 판마다 탐험률을 0.999배씩 감소 (10,000판 기준 넉넉한 감쇠율)

gamma = 0.99       
lr = 0.01      
batch_size = 32  
buffer_capacity = 1000


# 3. Model, Loss, Optimizer, Replay Buffer 생성 (변경 없음)
q_network = Network(n_states=n_states, n_actions=n_actions)
target_network = Network(n_states=n_states, n_actions=n_actions)
target_network.load_state_dict(q_network.state_dict())

optimizer = SGD(params=q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=buffer_capacity)



results: List[int] = []
# 4. Training Loop (DDQN 로직 적용)
for i in range(num_episode):
    current_state, _ = env.reset()
    all_reward = 0
    done = False 
    step = 0

    while not done and step < max_step:
        q_network.eval()

        # Epsilon-greedy 정책 (현재 exploration_prob 확률로 랜덤 행동)
        if torch.rand(1) < exploration_prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
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

            dones = dones.to(torch.float32) 
            
            q_values = q_network(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q_values_main = q_network(next_states)
                best_next_actions = next_q_values_main.argmax(dim=1, keepdim=True)  # (Batch, 1)

                next_q_values_target = target_network(next_states)  
                max_next_q = next_q_values_target.gather(1, best_next_actions).squeeze(1)   # best_next_action에 해당하는 Q-value들을 가져온다. 이후 squeeze(1)를 통해서 (Batch, 1) -> (Batch,)로 한차원 낮춰줘야한다.

                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # Loss 계산 및 업데이트 (Huber Loss)
            optimizer.zero_grad()
            loss = F.smooth_l1_loss(q_value, target_q) 
            loss.backward()
            optimizer.step()
            
        current_state = next_state
        step += 1

    # 에피소드 종료 후 타겟 네트워크 업데이트 
    target_network.load_state_dict(q_network.state_dict())
    
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
    

    if (i + 1) % 100 == 0:
        results.append(all_reward)
        if (i + 1) % 500 == 0:
            print(f"Episode {i + 1:5d} | Total Reward: {all_reward:.1f} | Epsilon: {exploration_prob:.3f}")

env.close()



# Results
episodes = [x * 100 for x in range(1, len(results) + 1)]

plt.figure(figsize=(10, 6))
plt.plot(episodes, results, linestyle='-', color='blue', alpha=0.7)

plt.title('DDQN Learning Curve - FrozenLake', fontsize=16, fontweight='bold')
plt.xlabel('Learning Step (Episode)', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.yticks([0.0, 1.0])
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('value_base/ddqn/learning_curve.png', dpi=300, bbox_inches='tight')

plt.show()