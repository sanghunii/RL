import numpy as np

TEST_NUM = 100

for i in range(TEST_NUM):
    # 총 100번의 test
    uniform_results = []    # 그냥 무작위로 선택 했을 때
    greedy_results = []     # 오차가 큰 값들만 골랐을 때
    greedy_is_results = []  # 오차가 큰 값들 위주로 뽑고 is_weights를 곱해줬을 때 

    # 1. 10,000개의 데이터와 Target 설정
    np.random.seed(42)
    N = 1_000_000
    data = np.random.randint(1, 5001, size=N)
    target = np.random.randint(1, 5001)

    # 2. 개별 데이터의 Loss 계산 및 실제 전체 평균 Loss (정답)
    losses = np.abs(target - data)
    true_mean = np.mean(losses)

    batch_size = 100

    # 3-1. Uniform Random Sampling: 무작위로 128개 추출
    uniform_indices = np.random.choice(N, size=batch_size, replace=False)
    uniform_losses = losses[uniform_indices]

    # 3-2. Greedy Sampling: Loss가 가장 큰 상위 128개 무조건 추출
    greedy_indices = np.argsort(losses)[-batch_size:]
    greedy_losses = losses[greedy_indices]

    # [중요] Greedy 방식에서의 가상의 '뽑힐 확률' 계산 
    total_loss_sum = np.sum(losses)
    greedy_probs = losses[greedy_indices] / total_loss_sum

    # 4. IS Weight 계산 (확률의 역수를 취하고, 데이터 개수 N으로 보정)
    is_weights = 1.0 / (N * greedy_probs)


    # 5. 결과저장 (true mean과의 오차값을 저장)
    uniform_results = abs(true_mean - np.mean(uniform_losses))
    greedy_results = abs(true_mean - np.mean(greedy_losses))
    greedy_is_results = abs(true_mean - np.mean(greedy_losses * is_weights))
    


print(f"TEST횟수: {TEST_NUM}\n"
      f"uniformly sampling 평균 오차값: {np.mean(uniform_results)}\n"
      f"only |error| 순서로 smapling 평균 오차값: {np.mean(greedy_results)}\n"
      f"|error|값 크기 우선 + Importance Sampling Weight 평균 오차값: {np.mean(greedy_is_results)}\n"
      )