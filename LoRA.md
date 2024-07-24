
## LoRA : Low-Rank Adaptation of Large Language Models

문제 정의 (언어 모델링 문제) : Task-specific prompt가 주어졌을 때 조건부 확률 최대화

가정

$P_{\Phi}(y|x)$ : a pre-trained autoregressive language model ($\Phi$ 는 파라미터)

$P_{\Phi}(y|x)$는 GPT와 같은 a generic multi-task learner일 수 있음

사전 학습된 모델 $P_{\Phi}(y|x)$를 실제 수행할 작업 '텍스트 생성 작업':(요약, MRC(기계독해), NL2SQL(자연어를 SQL 명령어로 변환))에 적용하면 각각의 작업들은 context-target 쌍으로 이루어진 학습 데이터셋 $Z={(x_i,y_i)}_{i=1,...,N}$으로 표현된다.
여기서, $x, y$는 모두 토큰들의 시퀀스이고, NL2SQL = (x:자연어 쿼리, y:일치하는 SQL 명령어), 요약=(x:기사 본문,y:기사 요약)와 같이 표현될 것이다.

일반적인 파인튜닝 (full fine-tuning)하는 동안, 모델은 사전학습한 가중치 $\Phi_0$로 초기화하고 weight를 $\Phi_0 + \Delta\Phi$ 식으로 업데이트할 것이다.

$$
max_{\Phi} \Sigma_{(x,y)\in Z} \Sigma_{t=1}^{|y|} log( P_\Phi (y_t | x,y < t) )
$$
