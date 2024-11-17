# $\epsilon-greedy$ algorithm for multi-armed problem 

### pseudo 
```
a = 1,2,......,k에 대해서 초기화 ##이때 a는 선택할 수 있는 action 항목들 
Q(a) = 0                     ##각 행동들에 대한 초기 가치 추정값은 0
N(a) = 0                     ##N은 각 행동들이 선택 당한 횟수.

while True: #무한루프
	A = 1-(epsilon)의 확률로 argmax_aQ(a) or (epsilon)확률로 탐험  #A는 그 시점에서 실제 선택한 행동 (활용 or 탐험)
	R <- bandit(A) ##bandit()은 A에 대한 이득 계산 함수
	N(A) <- N(A) +1 #행동 A가 선택된 횟수 
	Q(A) <- Q(A) + 1/N(A)[R-Q(A)] #A에 대한 Q값 갱신. 
```
