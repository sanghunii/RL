import torch 
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        
        dim1 = 256
        dim2 = 256 // 2
        dim3 = 256 // 4
        dim_head = 256 // 8
        
        # === 1. shared network ====
        self.shared = nn.Sequential(
            nn.Linear(self.n_states, dim1),
            nn.Mish(),

            nn.Linear(dim1, dim2),
            nn.Mish(),

            nn.Linear(dim2, dim3),
            nn.Mish()
        )
        
        # === 2. Value Head  === 
        self.value_head = nn.Sequential(
            nn.Linear(dim3, dim_head),
            nn.Mish(),

            nn.Linear(dim_head, 1)
        )

        # === 3. Advantage Head ===
        # '이 상태에서 각 행동(Action 조합)을 취하는 것이 얼마나 더 이득인가?'를 평가하므로 
        # 출력은 조합된 총 경우의 수인 n_actions개입니다.
        self.adv_head = nn.Sequential(
            nn.Linear(dim3, dim_head),
            nn.Mish(),

            nn.Linear(dim_head, n_actions)
        )


    def forward(self, state):
        shared = self.shared(state)
        value = self.value_head(shared)
        advs = self.adv_head(shared)

        return value, advs


    def get_q_values(self, state):
        """
        Aggregating Module
        
        순전파 과정에서 계산한 V값과 전체 Action(조합된 경우의 수)의 Advantage 값을 조합해서 Q-value를 산출한다.
        """
        value, advs = self.forward(state=state)

        # 1. 전체 Advantage value을 구하고 해당 평균값 이용해서 정규화
        advs_mean = advs.mean(dim=-1, keepdim=True)
        advs_normalized = advs - advs_mean      # (A(s,a)- E(A(s,a)))

        # 2. Q(s,a) = V(s) + (A(s,a) - E(A(s,a)))
        q = value + advs_normalized

        # 3. return
        return q