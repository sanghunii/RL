# Off-policy MonteCarloMethod for optimize target policy
## Use Weighted Importance Sampling for evaluating value of target policy

off-polcy MC는 기본적으로 두 개의 정책을 사용한다.

b : behavior policy로 해당 정책을 이용해서 에피소드를 생성하고 각 state에 대한 이득값을 도출해 낸다.

$\pi$ : target policy로 정책 b로 부터 얻은 이득값을 이용해서 정책 $\pi$ 의 가치를 계산한다. 

$\rho_{t:T(t)-1}$ : 중요도 추출 비율(importance sampling ratio)이다.이는 $\rho_{t:T(t)-1} = \Pi_{k=t}^{T(t)-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$와 같이 계산되며 
정책 b로부터 정책 $\pi$ 의 가치를 계산할때 해당 비율값을 이용해서 계산한다.



### pseudo
```
모든 s \in S, a \in A(s)에 대해 초기화:
	Q(s,a) \in R   ## (초기값은 랜덤하게 설정)
	C(s,a) = 0     ##(C_n은 n시점까지의 가중치들의 총합)
	\pi(s) = argmax_aQ(s,a)   ##정책 \pi는 행동 정책으로 부터 도출한 가치 추정값에 대해 탐욕적이다.

(각 에피소드에 대해)무한 루프:
	b = 임의의 소프트 정책 ##b는 행동 정책. 행동을 선택하는 정책으로 여기서 발생하는 이득값 데이터로 목표 정책의 가치를 추정한다.  
	정책 b를 활용하여 에피소드 생성 : S_0, A_0, R_1, S_1, A_1, R_2, . . . . ., S_{T-1}, A_{T-1}, S_T
	G = 0
	W = 1
	에피소드 각 단계에 대한 무한루프, t= T-1, T-2, . . . , 0:
		G = \gamma G+ R_{t+1}
		C(S_t, A_t) = C(S_t, A_t)+W
		Q(S_t, A_t) = Q(S_t, A_t) + W/C(S_t, A_t) * [G-Q(S_t,A_t)]
		\pi(S_t) = argmax_aQ(s,a)
		A_t \neq \pi(S_t)이면 루프 종료
		W = W*(1/b(A_t|S_t)
```
