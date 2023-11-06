# [KSC2023]맨하탄 거리와 체비셰프 거리를 활용한 클러스터링 기반 자기 지도 학습을 통한 이미지 표현 학습의 향상
맨하탄 거리와 체비셰프 거리를 활용한 클러스터링 기반 자기 지도 학습을 통한 이미지 표현 학습의 향상

# 요약 
:   이 연구에서는 자기 지도 학습(Self-supervised Learning) 방법 중 클러스터링을 사용한 Online Constrained K-Means (CoKe) 방식을 개선하여, 
단일 뷰만으로 이미지 표현(representation) 학습을 효과적으로 수행하는 새로운 방법을 제안한다. 
기존의 CoKe 방식은 유클리드 거리 계산에 의존하여 비슷한 이미지들에게 동일한 라벨을 부여하는 클러스터링 기반의 자기 지도 학습 방식이다. 
그러나 이 연구에서는 맨하탄 거리 및 체비셰프 거리를 활용하여 이미지 간의 유사성 측정 방식을 개선하였다. 
이렇게 변경된 거리 계산 메트릭은 일정 거리 이상으로 분리된 비슷한 이미지들에게 다른 라벨을 부여할 수 있게 하며, 이를 통해 이미지 표현 성능이 향상되었다. 
단일 뷰만으로 라벨링하는 과정에서 발생할 수 있는 어려움은 체비셰프 거리를 활용한 클러스터 중심 업데이트 메커니즘을 도입함으로써 극복하였다. 
이로 인해, 이상치의 영향력이 줄어들어 보다 안정적인 클러스터링 결과를 얻었다.

### 클러스터링 결과
- CoKe
  
![image](https://github.com/amthreeh/ksc2023-SSL_clustering/assets/103898937/d007868c-94c6-4437-b235-0745b5af2605)

- 맨하탄 거리[왼쪽] / 췌비셰프 거리[오른쪽]

 ![image](https://github.com/amthreeh/ksc2023-SSL_clustering/assets/103898937/6d81f38f-b491-4418-859e-687fcd610c36)




- Data:
1. CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
2. STL10: https://cs.stanford.edu/~acoates/stl10/
3. CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

# CoKe(유클리디언 거리 방식)
    - coke_v1_c10.py
    - coke_v1_stl10.py
    - coke_cifake.py
    
# 맨하탄 거리 방식
    - mh_v1_cifar10.py
    - mh_v1_stl10.py

# 체비셰프 거리 방식
    - cc_v1_stl10.py
    - cc_cifake.py
      
- 테스트(test_10.py)
