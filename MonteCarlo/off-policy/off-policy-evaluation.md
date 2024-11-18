# Off-policy MonteCarloMethod for estimate target policy value
## Use Weighted Importance Sampling for evaluating value of target policy
## 이 알고리즘은 target policy의 가치를 추정하기 위한 알고리즘 이다.

off-polcy MC는 기본적으로 두 개의 정책을 사용한다.

b : behavior policy로 해당 정책을 이용해서 에피소드를 생성하고 각 state에 대한 이득값을 도출해 낸다.

$\pi$ : target policy로 정책 b로 부터 얻은 이득값을 이용해서 정책 $\pi$ 의 가치를 계산한다. 

$\rho_{t:T(t)-1}$ : 중요도 추출 비율(importance sampling ratio)이다.이는 $\rho_{t:T(t)-1} = \Pi_{k=t}^{T(t)-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$와 같이 계산되며 
정책 b로부터 정책 $\pi$ 의 가치를 계산할때 해당 비율값을 이용해서 계산한다.



### pseudo
```
###target policy의 가치값을 추정하기 위한 비활성 정책 몬테카를로방법 제어

input = 임의의 target policy pi 
모든 s \in S, a \in A(s)에 대해:
	Q(s,a) = 임의로 설정
    C(s,a) = 0
  
(각 에피소드에 대해 무한루프):
	b = 정책 pi가 보증된 임의의 정책
    정책 b를 따르는 에피소드를 생성: S0,A0,R1,....., S_{T-1}, A_{T-1}, R_T
    G=0
    W=1
    에피소드의 각 단계에 대한 루프, t=T-1, T-2, .... ,0:
    	G = \gamma G + R_{t+1}
        C(S_t,A_t)=C(S_t,A_t) + W
        Q(S_t, A_t) = Q(S_t, A_t) + W/C(S_t,A_t)[G-Q(S_t,A_t)]
        W = W*pi(A_t|S_t)/b(A_t|S_t)
        W = 0이면 루프 종료 
  	
```
