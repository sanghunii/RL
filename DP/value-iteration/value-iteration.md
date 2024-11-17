# Value-Iteration 
Policy Iteration은 Policy Evaluation과 Policy Improvement의 반복 과정이다.

Policy  Evaluation과정에서는 Bellman enq를 기반으로한 가치 갱신이 많은 횟수가 반복 됨으로써 어떤 상태 $s$ 에 대한 추정값이 실제 상태 가치 값으로 수렴한다.

이는 매우 많은 컴퓨팅 자원과 시간을 요구한다. 이를 해결하기 위한 방법 중 하나가 Value-iteration이라는 알고리즘이다. 

Value-Iteration은 각 상태에 대한 가치값 갱신이 단 한번 일어난다. 

그대신 이전 방법과는 다르게 $average$ 값이 아닌 $max$ 값을 이용해서 상태의 가치를 갱신한다. 

아래는 기존의 가치 갱신식과 value-iteration algorithm의 가치 갱신 식이다. 차이점을 확인해 보라.

### Previous 
$v_{k+1}(s) = \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+ \gamma v_k(s')]$

### Value-Iteration
$v_{k+1}(s) = \underset{a}{max} \sum_{s',r}p(s',r|s,a)[r+ \gamma v_k(s')]$

### pseudo
```
오차값 : 추정의 정밀도를 결정하는 작은 기준값 \theta > 0
모든 s \in S+에 대해 V(s)를 임의로 초기화 단, V(종단)=0은 예외

루프 :
	\delta <- 0
	모든 s \in S에 대한 루프:
		v <- V(s)
		V(s) <- max_a\sum_{s',r}p(s',r|s,a)[r + \gamma V(s')]
		\delta <- max(\delta, |v-V(s))
		if \delta <= \theta면:
			루프종료
		
	이는 다음과 같은 경정론적 정책 \pi_*를 출력
```
