import gymnasium
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD 

# 모델 정의
class Network(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.linear = nn.Linear(in_features=self.n_states, out_features=self.n_actions)

    def forward(self, x):
        """
        이때 x는 input 즉, state이다.
        현재 내 상황은 frozen lake상황이 아니기 때문에 x = F.one_hot()을 이용해서 굳이 state를 vector로 변환시켜줄 필요 없이 그냥 torch.Tensor로 tensor로 변환시켜주면 된다."""
        x = torch.Tensor([x]) 
        x = F.one_hot(x.to(torch.int64), num_classes = self.n_states)
        x = x.to(torch.float32)
        y = self.linear(x)
        return y

# ENV 셋팅
env = gymnasium.make("FrozenLake-v1")

# 파라미터 셋팅
n_states = env.observation_space.n
n_actions = env.action_space.n
num_episodes = 100
exploration_prob = 0.01      ##탐험확률 
gamma = 0.99                
lr = 0.1
# 또한 우리는 Huber Loss를 사용하기 때문에 그에따른 임계값 delta또한 정해 줘야한다.

# 모델생성, LossFunction생성, Optimizer생성
network = Network(n_states=n_states, n_actions=n_actions)   ##create model 이건 그대로
criterion = nn.MSELoss()                                    ##Loss function 만들기 / 이거 연구구현할때는 내가 customizing해야함.
optimizer = SGD(params=network.parameters(), lr=lr)         ##이건 그대로





## ------------ ************** Train Code *************** --------------------


# train code ******여기가 중요 *******
for i in range(num_episodes):
    done = False                # Episode종료 여부 초기화
    all_reward = 0              # 누적 보상 초기화
    
    current_state, current_prob = env.reset()   # 초기 상태값 얻기 
    
    while not done:     # 각 에피소드별 학습 코드
        # network gradient 초기화
        optimizer.zero_grad()       
        
        # 현재 step에 대한 각 Q값 얻기 (아마 확률값으로 주어질듯 하다)
        Qs = network(current_state)
        
        # epsilon-greedy rule에 따라서 action선택
        if torch.rand(1) < exploration_prob:
            action = env.action_space.sample()     #여기서 .sample()메서드가 아마 무작위로 하나 가져오는것같다.
        else:
            action = torch.argmax(Qs).item()

        # 고른 action을 토대로 next_state, reward, done여부 를 받는다.
        next_state, reward, done, _ , _ = env.step(action=action)

        if done:    #에피소드가 끝났으면
            Qs[0, action] = reward
        else:
            # Target value 계산 
            Qs_next = network(next_state)       #next state에 대한 Q값들 받아옴 (loss 계산을 위해서)
            Qs[0, action] = reward + gamma*torch.max(Qs_next).item()

            # Predict value 계산 
            Qs_pred = network(current_state)

            # Loss 계산 
            loss = criterion(Qs, Qs_pred)       # 각 Q값에 대한 손실값 계산
            loss.backward()                     # 각 Q값의 손실값을 이용해서 각 Q값에 대한 Gradient계산

            # update params
            optimizer.step()                    # 각 Q값의 손실값 
            
        # state transtion & 보상값 추적 (all_reward)에 누적시킨다. 
        current_state = next_state
        all_reward += reward

        print("Loss: ", loss.item(), "Reward: ", reward, "done: ", done)