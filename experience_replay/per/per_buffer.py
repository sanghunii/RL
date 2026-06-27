import numpy as np
from typing import Tuple, List
import random
import torch

from experience_replay.per.sumtree import SumTree

# IS(Importance Sampling)을 이용해서 Priority base sampling으로 인해 발생하는 Bias를 해결해야 한다.
# alpha값이랑 beta값 어떻게 annealing할지 정해야함. -> Linear Annealing 적용


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha=0.6):
        """
        Args:
            capacity: Size of replay buffer
            alpha: 
                [0, 1]사이의 값을 가진다.
                sampling을 할 때 priority를 얼마나 반영할지. alpha in [0, 1].  if alpha = 0: random uniform,  else if plah->1.0: sampling시에 priority더 반영
        
        Explain:
           method: proportion
           arg set in origin thesis:
                alpha: 0.6 fix 
                beta: 0.4 -> 1.0 linearAnnealing
        """

        self.tree = SumTree(capacity=capacity)
        self.alpha = alpha      # Priority의 영향력 조절 (alpha = 0: Uniform Random Sampling)
        
        # 새 데이터가 들어올 때 '가장 높은 우선순위'를 부여하기 위한 기준값
        self.max_priority = 1.0 
        

    def push(self, state: Tuple[float], action: List[int], reward: int, next_state: Tuple[float], done: bool):
        """data 저장 (새 데이터는 항상 최대 우선순위로 지정하여 한 번은 학습되게 한다.)"""
        data = (state, action, reward, next_state, done)

        # p^alpha를 우선순위로 저장
        # 새 데이터는 현재까지의 max_priority를 주어 적어도 한 번은 학습되도록 합니다.
        p = self.max_priority 
        self.tree.add(p=p, data=data)


    def sample(self, batch_size, beta, device="cpu"):
        """
        Args:
            batch_size: 샘플링할 데이터 개수
            current_step: 현재 학습의 global step (beta annealing에 사용)
        """
        batch_idx, batch_priorities = [], []
        batch_data = []

        # IS 가중치 계산을 위한 준비 ->Batch Size만큼 구간을 나눈다. ;  batch_size = 128이라면 구간의 갯수는 총 128개 이고 한 구간당 길이는 sum_of_priority / batch_size가 된다. 
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, p, data = self.tree.get(s)     # idx: tree의 index, p: priority,  data: actual data (transition)
            batch_idx.append(idx)
            batch_priorities.append(p)
            batch_data.append(data)
        
        # Unpacking
        state, action, reward, next_state, done = zip(*batch_data)

        # Computing Importance Sampling weight   is_weights = (1/N * 1/P(i))^beta = (N * P(i))^-beta
        sampling_probabilities = np.array(batch_priorities) / self.tree.total_priority()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()      # Normalization

        return (
            torch.as_tensor(np.array(state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(action), dtype=torch.long, device=device),
            torch.as_tensor(np.array(reward), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(done), dtype=torch.bool, device=device),

            # Importance Sampling Weights
            torch.as_tensor(np.array(is_weights), dtype=torch.float32, device=device),
            # Tree Indices (나중에 update 시 리스트로 처리하므로 그대로 반환)
            batch_idx
        )
    

    def batch_update(self, tree_idx, abs_errors):
        """학습 후 계산된 TD-Error를 바탕으로 우선순위 갱신"""
        abs_errors += 1e-5      # 0이 되어 샘플링 안되는 것 방지
        ps = np.power(abs_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            self.max_priority = max(self.max_priority, p)


    def get_state(self):
        """저장을 위해 모든 상태 정보를 추출"""
        return {
            "tree": self.tree.tree,
            "data": self.tree.data,
            "write": self.tree.write,
            "n_entries": self.tree.n_entries,
            "max_priority": self.max_priority
        }
    
    def load_state(self, state_dict):
        """저장된 상태 정보를 불러와서 복구"""
        self.tree.tree = state_dict["tree"]
        self.tree.data = state_dict["data"]
        self.tree.write = state_dict["write"]
        self.tree.n_entries = state_dict["n_entries"]
        self.max_priority = state_dict.get("max_priority", 1.0)