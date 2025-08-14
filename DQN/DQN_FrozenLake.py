import gymnasium
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
from collections import deque
import random


# Network 정의 
class Network(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.linear = nn.Linear(in_features=n_states, out_features=self.n_actions)
        """self.linear에 input feature랑 output feature를 굳이 이렇게 안해도 될듯?
        또한 nn.Linear말고 Conv1d사용하는 것도 한번 고려해 보자."""

    def forward(self, x):
        x = torch.Tensor(x) if not torch.is_tensor(x) else x
        if x.dim() == 1:        ##Tensor.dim()은 Tensor의 차원 수를 리턴한다.
            x = x.unsqueeze(0)      ##unsqueeze()는 파라미터에 해당하는 index에 새로운 차원을 추가한다. 
        x = F.one_hot(x.to(torch.int64), num_classes=self.n_states).to(torch.float32)
        y = self.linear(x)
        return y


# Replay Buffer 정의 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)        # deque : double-ended que
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))   # deque.append()는 list.append()와 동일하게 작동한다. 
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) # Replay buffer에서 random sampling 이후 unpacking해서 각 요소들끼리 모은다. 
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) # doen즉 종료 여부는 빼도 되지 싶다. => 어짜피 한 에피소드 기준으로 학습을 시킬 것이다.
    


# ENV 및 Parameter 셋팅
env = gymnasium.make('FrozenLake-v1')   # 우리는 Custom ENV class를 들고오면 된다.
n_states = env.observation_space.n  # 그냥 input state의 크기를 넣으면 될것같다. 우리는 6 * 2 + 1 해서  13크기의 observatino space를 갖는다.
n_actions = env.actions_space.n # 그냥 Ouptut 갯수임. 
num_episode = 1 # 에피소드 갯수
max_step = 20   # 에피소드의 길이 (만약 전체 job이 20개면 아마 40.. ? )
exploration_prob = 0.5  # 탐험 확률 (보통 epsilon으로 표현)
gamma = 0.99        # 감쇠율 (Target Value구할 때 사용함)
lr = 0.1        # 학습률 (learning rate)
batch_size = 5      # leanring batch size / loss function계산할때 batch_size필요할듯?
buffer_capacity = 1000  # Replay Buffer의 최대길이
delta = 1.0     # Huber loss를 사용할 때 사용되는 임계값 


# Model, Loss, Optimizer, Replay Buffer 생성
q_network = Network(n_states=n_states, n_actions=n_actions) # q_network ; generate predict value 
target_network = Network(n_states=n_states, n_actions=n_actions)    #target_network ; generate target value 
target_network.load_state_dict(q_network.state_dict())  # 초기 가중치 동기화 
criterion_mse = nn.MSELoss()    # L2 Loss = MSE.
criterion_mae = nn.L1Loss()     # L1 Loss = MAE.  우리는 huber loss를 사용할 것 이므로 L1과 L2둘 다 필요함 
optimizer = SGD(params=q_network.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=buffer_capacity)



# Training Loop
for i in range(num_episode):    # num_episode만큼 생성해놓고 루프돌릴지 아니면 num_episode만큼 생성해놓고 학습 루프를 돌릴지 고민좀 해보자
    current_state, _ = env.reset()      # env.reset()에서 current state만 반환하면 되지 않을까? => current state만 반환시켜도 충분할듯 하다. 
    all_reward = 0
    done = False 
    step = 0

    while (not done & step < max_step):    # 이때 max_step은 한 에피소드에 나올 수 있는 최대 step 수
        # Action 선택 (epsilon-greedy)
        if torch.rand(1) < exploration_prob:
            action = env.action_space.smaple()  # 이때 우리는 2개의 action을 선택해야 한다. (mc1에서의 dispatching rule, mc2에서의 dispatching rule)
        else:
            with torch.no_grad():           # 여기서는 model.eval()과정을 거치지 않고 바로 데이터 모으기에 들어갔지만 (어짜피 network에 batch norm이나 drop out layer가 없어서) 내 실험 구현단계에서는 model.eval()코드가 앞에 선행되어야 한다.
                Qs = q_network(current_state)
                action = torch.argmax(Qs).item()
        
        # Environment step
        next_state, reward, done, _ = env.step(action=action) # 이때 action은 List[int, int]혹은 Tuple[int, int]로 주어진다. [mc1 action, mc2 action], 또한 마지막 return 값은 info인데 난 굳이 이거 구현 안할것이다.
        all_reward += reward

        # Replay buffer에 저장.
        replay_buffer.push(current_state, action, reward, next_state, done)

        # Replay buffer에서 샘플링 및 학습
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=batch_size)

            states = torch.tensor(states, dtype=torch.float64) 
            actions = torch.tensor(actions, dtype=torch.int64)  # action을 왜 tensor로 변형하려는지 잘 모르겠음
            rewards = torch.tensor(rewards, dtype=torch.float64)    # rewards를 왜 tensor로 변형하려는지 잘 모르겠음
            next_states = torch.tensor(next_states, dtype=torch.float64)    # next_state를 왜 tensor로 변형하려는지 잘 모르겠음
            dones = torch.tensor(dones, dtype=torch.int16)  # dones를 왜 tensor로 변형하려는지 잘 모르겠음
            
            # Q_value 계산 (for Predict Value => Q(s,a; theta))
            q_values = q_network(states)    # 모든 Q값들 다받아온다. shpe(q_network(states)) = (batch_size, n_actions)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # 각 state에서의 Q값 중에서 선택된 action의 Q값을 가지고옴
            
            # Target Q값 계산 (for Target Value => Q(s', a'; theta-))
            with torch.no_grad():
                next_q_values = target_network(next_states)     # 각 next_state에 대한 Q값 가져옴
                max_next_q = next_q_values.max(1)[0]            # 각 next_state에서의 max Q값 가져옴
                target_q = reward + gamma * max_next_q*(1 - dones)  # 만약 done이면 Target Value는 Reward로 한다.
            
            # Loss 계산 및 업데이트  - Huber Loss 구현
            optimizer.zero_grad()
            if ((np.average(np.array(target_q - q_value))) < delta):    # Loss가 delta(임계값 보다 작으면 => MSE)
                loss = criterion_mse(q_values, target_q)        
            else:
                loss = criterion_mae(q_values, target_q)
            
            print(f"Step: {step}, Loss: {loss.item():.4f}, Reward: {reward}, Done: {done}")

        current_state = next_state # State Transition
        step += 1

    # 에피소드 종료 후 타겟 네트워크 업데이트 
    target_network.load_state_dict(q_network.state_dict())
    print(f"Episode {i + 1}, Total Reward: {all_reward}")

env.close() # 환경을 초기화하면서 종료.