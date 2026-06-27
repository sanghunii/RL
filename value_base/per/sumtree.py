# Prioritized Experience Replay Buffer 
import numpy as np
import random

class SumTree:
    """
    SumTree는 Binary Tree의 일종이다.

    Leaf Node의 범위는 항상 [capacity-1, 2*capacity-2]이다.
    
    SumTree에서 난수를 뽑아서 데이터를 뽑을 때 아래의 규칙을 따른다:
        1. 난수하나 뽑는다.
        2. 뽑은 난수가 [0, 왼쪽에 있는 자식노드]의 범위에 속하는지 본다. 
        3. 속한다면 왼쪽으로 간다. 
        4. 속하지 않는다면 오른쪽으로 가면서 (난수 - 왼쪽 자식노드의 값)을 해준다.                          (즉, 왼쪽으로 갈 때는 값을 그대로 들고가고 오른쪽으로 갈때는 왼쪽 자식 노드의 값을 빼준다.)
        5. 리프노드라면 해당 리프노드의 값이 Sampling되는 값 그렇지 않는다면 2번으로 돌아가서 다시 수행

        e.g. [1,5,9,7]일때 8.3의 난수값을 얻었다면 Sampling되는 값은 9이다.
    """
    def __init__(self, capacity):
        self.capacity = capacity

        # 트리 전체 노드 수 : 2 * capacity - 1
        # Actual Priority is saved on leaf node 

        self.tree = np.zeros(2 * capacity - 1)          # priority를 저장
        self.data = np.zeros(capacity, dtype=object)    # An array to contain actual transition (leaf nodes). range = [capacity-1, 2*capacity-2]
        self.write = 0                                  # where to write data. range = [0, capacity - 1].     write attribute always point leaf nodes
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        부모노드로 올라가며 차이값을 더해준다. 
        시간복잡도 : O(logN)

        update() method에서 사용
        """
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
        
    def _retrieve(self, idx, s) -> int:
        """
        난수 s가 속한 리프 노드 인덱스 찾기. 시간복잡도: O(logN)

        get() method에서 사용
        """
        
        # 1. 현재의 부모노드 기준으로 자식노드들의 index계산.
        left = 2 * idx + 1     
        right = left + 1

        # 2. 현재 노드가 leaf node이면 현재 node의 index를 return
        if left >= len(self.tree):
            return idx 
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total_priority(self):
        """현재 Leaf Node에 저장되어 있는 Priority의 합을 반환한다."""
        return self.tree[0]

    
    def update(self, idx, p):
        """
        기존 데이터의 우선순위 업데이트. O(log N)

        idx: tree의 idx 
        p: priority 
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change=change)
    
    def add(self, p, data):
        """
        새로운 데이터와 초기 우선순위 추가
        """
        idx = self.write + self.capacity - 1        # BinaryTree에서 leaf node의 index = [capacity - 1, 2*capacity - 2]
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def get(self, s):
        """
        샘플링을 위한 데이터 추출
        """
        idx = self._retrieve(0, s)              # tree의 idx. tree에는 priority가 저장된다.
        data_idx = idx - self.capacity + 1      # data의 idx. data에는 실제 transition들이 저장된다.

        return (idx, self.tree[idx], self.data[data_idx])



def main():

    import numpy as np

    # 1. 초기화 (저장 공간이 4개인 트리 생성)
    capacity = 4
    st = SumTree(capacity)


    # 2. 데이터 추가 (add)
    # (우선순위, 실제 데이터) 순서로 입력합니다.
    # 처음에는 보통 최대 우선순위(예: 1.0)를 주어 모든 데이터가 한 번은 학습되게 합니다.
    print("--- 데이터 추가 단계 ---")
    experiences = [("state1", "action1"), ("state2", "action2"), ("state3", "action3"), ("state4", "action4")]
    for exp in experiences:
        st.add(1.0, exp)


    print(f"전체 우선순위 합 (Root): {st.total_priority()}") # 1.0 * 4 = 4.0 이 나옴


    # 3. 데이터 샘플링 (get)
    # PER에서는 0부터 Total Priority 사이의 난수를 뽑아 해당 구간의 데이터를 가져옵니다.
    print("\n--- 데이터 샘플링 단계 ---")
    s = 2.5  # 0 ~ 4.0 사이의 난수라고 가정
    idx, priority, data = st.get(s)
    print(f"뽑힌 난수: {s}")
    print(f"트리 인덱스: {idx}, 우선순위: {priority}, 데이터: {data}")


    # 4. 우선순위 업데이트 (update)
    # 학습 후 TD-error가 계산되면 그에 맞춰 우선순위를 갱신합니다.
    # 예: state2(인덱스 4 혹은 5 부근)의 오차가 커서 우선순위를 5.0으로 올리고 싶을 때
    print("\n--- 우선순위 업데이트 ---")
    # 실제 사용 시에는 get()에서 받아온 idx를 그대로 사용합니다.
    new_priority = 5.0
    st.update(idx, new_priority) 

    print(f"업데이트 후 전체 우선순위 합: {st.total_priority()}")

if __name__ == '__main__':
    main()
