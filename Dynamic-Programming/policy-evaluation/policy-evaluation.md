 # Policy Evaluation Algorithm for Dynamic Programming 

 ### pseudo
```
#pseudo code
평가 받을 정책 pi가 입력으로 들어간다.
추정의 정확도 판단은 theta를 이용한다.
모든 상태에 대해 v(s)의 초기값은 random. 다만 terminal state의 value = 0 으로 한다.

while True:
	delta <- 0 ##초기 오차값 지정
	모든 s에 대한 루프:
		v <- V(s)
		V(s) <- \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]
		delta <- max(delta, |v-V(s)|)  ##새로운 오차값과 기존의 오차값중 큰 것을 사용
    이 루프를 delta < theta를 만족할때까지 반복 
```
