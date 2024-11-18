# On-policy MonteCarloMethod

활성정책 제어방법은 하나의 정책을 사용한다.

즉 데이터를 생성하는 정책과 최적화의 목적이 되는 정책이 동일하다.

활성 정책 제어 방법에서는 일반적으로 soft한 정책을 사용한다.

여기서 정책이 soft하다는 것은 모든 $a \in A(s)$ 에 대하여 $pi(a|s) > 0$ 을 만족하지만 정책이 결정론적인 최정 정책 방향으로 나아감을 의미한다.

여기서는 아래와 같이 정책이 정의된다.

$$\pi(a|s)=\begin{cases}1-\epsilon+\frac{\epsilon}{|A(s)|} (for-greedy-action) \\
\frac{\epsilon}{|A(s)|} (for-not-greedy-actions)\end{cases} $$


### pseudo
```
알고리즘 파라미터 : 작은 양의 값 epsilon > 0
initialize:
	pi = 임의의 epilon-soft policy
    모든 s \in S, a \in A(s)에 대한 Q(s,a)임의값으로 초기화
    Returns(s,a) = 모든 s \in S, a \in A(s)에 대해 빈 리스트 
   
 각 에피소드에 대해 무한 루프:
 	pi를 따르는 에피소드 생성 : S0,A0,R1,S1,A1,R1,S2,A2,....,S_{T-1}, A_{T-1}, R_T
    G = 0
    에피소드 각 단계에 대한 루프, t=T-1, T-2, .... , 0:
    	G = \gamma G + R_{t+1}
        S_t, A_t 쌍이 S0,A0,.....,S_{t-1},A_{t-1}중 나타나지 않는다면:
        	G를 Return(S_t, A_t)리스트에 추가
            Q(S_t, A_t) = average(Returns(S_t,A_t))
            A* = argmax_aQ(S_t,a)) (이때 조건을 만족하는 a가 두개 이상이면 둘 중 무작위로 선택) 
            모든 a \in A(S_t)에 대해:
            	pi(a|S_t) = (if a is A*: 1-epsilon+epsilon/|A(S_t)|) or (if a is not A*: epsilon/|A(S_t)|)
```
