# explaninableAI
XAI

딥러닝 모델은 수많은 비선형의 조합으로 이루어져 핵심적인 Task의 판단을 어떻게 하는지 설명되기 어렵다.

따라서, 딥러닝 모델이 잘 구성되었는지 설명하기위해 Explaninable AI 기법이 필요하다.

어느 그림에서 어떤 부분이 Task 결과에 영향을 미치는지 파악하기 위한 척도이다.

한 예로, CNN에서 다양한 Layer의 Conv units은 pixel 단위로 특징을 표현할 수 있다. 

하지만, fully connected layer를 적용하면 Flatten vector로 변환되어 object localize하는 특징을 잃어버린다. 

그래서, 이러한 문제를 해결하기위해 Global average pooling을 적용한 Class activation mapping 기술이 고안되었다.

즉, 각 conv layer의 feature를 map으로서 visualization을 가능하게했다.

하지만, 여기서 궁금증이 생긴다. 이 feature를 시각화하고 핵심 region을 찾는다고 해서 내가 할 수 있는 것은 무엇일까?

어느 label에 대한 표현이 핵심 region을 가지고 있겠지만, 그 부분을 강화시키거나 변환해버려야 하는 것인가?

그럼, layer 마다 다를것이고 좀 더 덜 필요한 layer가 있고 핵심 layer가 있을 것인데 그것을 아예 배제를 하는게 맞을 것인가?

근데 배제를 한다면, 그 layer가 없어짐으로서 발생하는 side effect가 있지 않을까?

------------------------------------------------------------------------------------------------------------------------

1. [Class Activation Map(CAM), https://arxiv.org/pdf/1512.04150.pdf]: Global Average Pooling Layer 전의 feature map의 parameters와 Global Average Pooling Layer 후의 FC Layer의 Weight를 dot product한 Class Activation Map을 생성하여 Classification 예측 결과를 출력하기 전에 가장 정답에 가까운 Class는 이미지 상에서 어느 영역에서 핵심적인 정보를 보고 판단하는지 시각화 할 수 있음.

2. [Grad CAM, https://arxiv.org/pdf/1610.02391.pdf]: 종단의 activation function을 적용하기 전 FC Layer가 아닌 Global Average pooling이 적용된 모델에 한해서만 CAM이 수행되었지만, 이를 극복하여 Fully connected layer 구조의 모델도 시각화가 가능하도록 연구됨.



