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
n_actions = env.action_space

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
    global steps_done
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

# TODO
"""함수 내에 globa로 선언하면 이거 어떻게 사용할 수 있는지 그리고 select_action()함수가 어떻게 작동하는지 알아보고 넘어가기 """