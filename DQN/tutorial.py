import gymnasium as gym
import math 
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count 

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# set for matplotlib 
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# set for use gup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




"""Replay Memory"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 지정"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.smaple(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) ##input layer
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호출.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x) # ([[left0ext, right0exp]])



# BATCH_SIZE는 리플레이 버퍼에서 샘플링된 트랜지션의 수
# GAMMA는 이전 섹션에서 언급한 할인 계수이다.
# EPS_START는 epsilon 시작 값
# EPS_END는 epsilon의 최종 값
# EPS_DECAY는 epsilon의 exponential decay속도를 제어, 높을수록 감쇠 속도가 느리다.
# TAU는 목표 네트워크의 업데이트 속도
# LR is learning rate of optimizer for 'AdamW'
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# gym 행동 공간에서 행동의 숫자를 얻는다.
n_actions = env.action_space.n

# 상태 관측 횟수를 얻는다. 
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device=device)
target_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device=device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done #steps_done은 함수 밖에서 선언된 전역변수
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환한다.
            # 최대 결과의 두 번째 열은 최대 요소의 주소값이므로, 
            # 기대 보상이 더 큰 행동을 선택할 수 있다.
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    

episode_duration = []

def plot_durations(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_duration, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# 최종적으로 모델 학습을 위한 코드.
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    ##replay memory에서 학습을 위한 data를 꺼낸다.
    transitions = memory.sample(batch_size=BATCH_SIZE) 
    batch = Transition(*zip(*transitions))  ## 이거 어떻게 돌아가는지 작동방식 잘 설명할 수 있는지 스스로 체크

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결
    # (최종 상태는 시물레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None,
                                            batch.next_state)), device = device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t,a)계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1).values로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values()

    #기대 Q값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화 
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()