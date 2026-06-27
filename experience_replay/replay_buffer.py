import torch
import random
from collections import deque
from typing import Tuple, List



# Replay Buffer 정의 
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)


    def __len__(self):
        return len(self.buffer)
    
    def push(self, state: Tuple[float], action: int, reward: float, next_state: Tuple[float], done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, device = "cpu"):
        """state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) # Replay buffer에서 random sampling 이후 unpacking해서 각 요소들끼리 모은다. 
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) # doen즉 종료 여부는 빼도 되지 싶다. => 어짜피 한 에피소드 기준으로 학습을 시킬 것이다."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.bool, device=device)
        )    
    
    def sample_all(self):
        if len(self.buffer) == 0:
            raise Exception("Replay buffer가 비어있습니다.")
        
        return self.buffer


    def clear(self):
        """
        replay buffer를 비운다.
        """
        self.buffer.clear()