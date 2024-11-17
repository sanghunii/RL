# Policy Iteration for Dynamic programming

### process
1. Policy Evaluation -> find state value $v_\pi(s)$ for each state
2. Policy Improvement -> based on found value $v_\pi(s)$, improvement policy
3. repeat.


### pseudo
```
##pseudo code##

1. Initiation
 모든 s \in S에 대해 임의로 v(s) \in \R와 \pi(s) \in A(s)를 설정
 
 
 2. Policy Evaluation ##현재 정책 \pi에 대한 최적 상태 가치 함수의 근사값을 구한다. 
  루프:
	  오차값 <- 0
	  모든 s \in S에 대한 루프:
		  v <- V(s)
		  V(s) <- \sum_{s',r}p(s',r|s,\pi(s))[r+\gamma V(s')]
		  
		  
3. Policy Improvement ##새로운 정책 \pi'의 가치를 계산하고 이전 정책 \pi의 가치와 비교 
	안정적 정책 <- true
	모든 s \in S에 대해:
		이전행동 <- \pi(s)
		\pi(s) <- argmax_a\sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]
		if 이전행동 =/ pi(s):
			2번으로 돌아가서 새로운 정책에 대한 가치값 계산하고 3번루프를 다시 수행
		else 이전행동 == pi(s):
			현재 정책이 최적 정책이므로 루프를 종료하고 가치값과 정책을 반환	
```
