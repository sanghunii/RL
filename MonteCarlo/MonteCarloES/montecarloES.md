# MonteCarloES 

ES means ExploratoryStart 

결정론적 정책 $\pi$ 를 따르고 있다고 가정해 보자.

이때 문제점이 어떤 상태에 대하여 agent가 선호하는 행동만을 계속 선택하게 되어 어떤 상태-행동 쌍은 선택 자체가 되지 않는다.

이를 해결하기 위해 정책이 평가와 향상을 반복하면서 최적 정책 $\pi_*$ 으로 수렴함을 보장하기 위해 모든 상태-행동 쌍에 대하여 시작 단계에서 선택될 확률로 동일한 확률을 부여 하였다.

이렇게 되면 에피소드를 무한번 반복했을때 모든 상태-행동 쌍이 무한번 선택 됨으로써 정책 $\pi$ 의 최적 정책 $\pi_*$ 으로 수렴을 보장한다.

하지만 이 아이디어는 현실에서는 적용하기 어려우며 결국에 우리가 할 수 있는 것은 결정론적 정책이 아닌 확률론적 정책만을 고려하는 것이다.

### pseudo
```
##MonteCarloES algorithm
##Use First-visit MC method for evaluating policy 

초기화: 
	모든 s에 대하여 정책 pi(s)를 임의로 설정 
	모든 (s,a)에 대하여 Q(s,a)를 임의로 설정
	모든 (s,a)에 대해 빈 리스트를 Returns(s,a)에 대입

(각 에피소드에 대해)무한 루프:
	모든 (s,a)쌍에 대한 확률이 0보다 크도록 설정하고 무작위로 하나의 쌍을 선택
	정책 pi에 따라 선택한 쌍 (s0, a0)으로 부터 에피소드 생성: s0,a0,r1,s1,a1,r2,....
	G = 0
	에피소드 각 단계에 대한 무한루프, t=T-1,T-2,T-3,....,0:
		G에 \gamma G + R_{t+1}대입
		S_t, A_t쌍이 S0,A0,.....,S_{t-1},A_{t-1}에 없으면: ##First-Visit이 맞는지 확인
			G를 Returns(S_t,A_t)에 추가
			argmax_a Q(S_t,a)를 \pi(S_t)에 대입
```
