# ML-HW-Fall-2023

## 과제 개요

Numpy를 이용하여 Neural Network를 구성하고 MNIST data에 적용하여 평가한다.
Numpy로 구성한 Neural Network와 동일한 구성의 Neural Network를 PyTorch를 이용하 여 만들고 결과를 비교한다.

## 구현 환경 OS : Windows 10

Hardware: intel 8th i5 CPU, not using GPU

Integrated Development Environment : Visual Studio Code, python 3.11.4

## 알고리즘에 대한 설명
  Neural Network는 딥러닝에서 가장 기본이 되는 구조로 뉴런(두뇌의 신경세포)이 연결된 형
  태를 모방한 모델이다.
  인간의 두뇌는 뉴런들이 복잡하게 연결된 네트워크를 형성되어있다. 이 뉴런들은 인간이 정보 를 처리하는 데 도움이 되도록 서로 전기 신호를 보낸다. 마찬가지로 Neural Network는 문 제를 해결하기 위해 함께 작동하는 인공 뉴런으로 구성된다. 아래 그림은 뉴럴 네트워크의 한 예시로 인공 뉴런은 아래 그림에 있는 원 모양의 노드라고 하는 소프트웨어 모듈로 구성된다.
  
  ![image](https://github.com/user-attachments/assets/1334350f-97b9-4ef8-b90b-3df0be411144)

  기초적인 설명은 간략히 마치고 과제에서 사용한 기법에 대해 설명한다.
  기본적으로 계층은 affine layer와 활성화 함수 계층을 사용하였다. 활성화 함수 계층으로는 최근 자주 사용되는 ReLU와 LeakyReLU, SiLU를 구현하였는데, 3개의 계층을 모두 사용하 여보았을 때 확연한 차이를 보이지는 않았다. 과제에서는 평균적으로 가장 좋은 결과를 보인 LeakyReLU를 활성화 함수 계층으로 채택하였다. output layer의 출력 크기는 10으로 설정 하여 [0~9]까지의 확률값을 반환하도록 하였다. 이 때 Loss layer로 softmax with loss 계층 을 사용하고 손실함수로 cross entropy error를 사용하여 분류 모델을 구현하였다. Loss layer로 mean square error loss도 구현하여 output 크기를 1로 하고 출력값을 반올림하여 target을 찾는 회귀 형식의 모델도 구현해보았는데 아래와 같이 소수점 차이로 예측이 빗나가 는 경우가 많아 cross entropy error를 이용한 분류 모델로 구현하였다.

  ![image](https://github.com/user-attachments/assets/ccb82466-b891-4c37-ad13-35ac5e1bd88d)

    
  affine 계층과 활성화함수 계층만 사용하였을 때 loss값이 nan이 나오는 경우가 생겨 학습이 제대로 이루어지지 않았기 때문에 활성화값의 분포에 문제가 있다고 생각되어 Batch Normaliztion 계층을 구현하여 아래 그림처럼 affine 계층과 활성화 함수 계층 사이에 위치 하도록 구성하였다. (과제의 경우 그림의 Relu 계층을 Leaky ReLU 계층으로 대체)

  ![image](https://github.com/user-attachments/assets/469f20ab-fc76-4b67-805e-2ddd8e31d468)

  또, Affine계층의 가중치는 He initialization을 채택하여 초기화함으로써 Leaky ReLU( 및 ReLU)에 특화된 초기화 값을 사용하고 활성화값의 분포를 균일하게 하였다.
  마지막으로 학습에 사용되는 optimizer로 SGD과 최근 자주 사용되는 Adam을 구현하였다. SGD는 역전파법을 이용하여 구한 가중치의 미분값에 학습률을 곱하여 최적의 값을 찾는 optimizer이다. SGD는 단순히 기울어진 방향으로 최적화가 진행되므로 탐색 경로가 비효율적 이라는 단점이 있다. Adam은 빠르게 이동해야 하는 축으로는 더 빠르게 이동하는 Momentum(gradient의 지수가중평균을 계산(first moment m))과 학습률을 동적으로 조절 하는 RMSProp(gradient의 제곱에 대한 지수가중평균을 계산(second moment v))의 개념을 합쳐 SGD의 단점을 해결하였다.
  batch normalization과 adam의 구현은 5절에서 설명한다.

## 데이터에 대한 설명
### Input Feature
Mnist 데이터는 숫자 0~9에 해당하는 손글씨 이미지 데이터이다.
이미지는 28*28 pixel로 이루어져 있으며 traing data로 60000개, test data로 10000개 data로 구성된다. 입력 데이터는 flatten하여 28*28 크기를 784로 만들어 입력한다.
또, input feature의 최대값인 255.0로 나누어 모든 feature를 0~1사이값으로 normalize한다.
즉, train data의 input은 (60000 * 784), test data의 input은 (10000 * 784)의 shape를 가진다
코드는 dataset의 mnist.py 참고
### Target Output
입력된 이미지의 class를 [0 ~ 9]사이 값으로 예측하고 실제 정답과 비교하여 입력된 총 데이터에 대한 정확도를 출 력한다. target data는 정답에 해당하는 label의 index값만을 1로 하는 one-hot labeling을 사용하여 shape를 (data 개수 * 10)으로 한다.
코드는 dataset의 mnist.py 참고
 
## 소스코드에 대한 설명
### Neural Net Using Numpy - init
![image](https://github.com/user-attachments/assets/1cb055ad-aa04-4e42-9a06-640b8b588dae)

다음과 같이 hidden layer가 4개인 neural net을 구성하였다. 코드의 hidden_size는 256으로 hidden layer의 unit 수는 입력층에서 가까운 순서부터 256, 128, 64, 32이다. 마지막 hidden layer의 affine 계층의 Weight는 (32*10)으로 (10*batch 크기)를 output으로 내놓는다.
각각의 layer는 NN_Numpy_layers.py 파일에서 확인할 수 있다.
- BatchNorm class
NN_Numpy_layers.py 의 기본적인 layer 설명은 생략하고 batch_norm layer를 설명한다. batch-normalizatin을 수식으로 나타내면 다음과 같다. 평균과 분산을 구하고, 입력을 정규화 시켜준다.
정규화 과정에서 평균을 빼주고 그것을 분산으로 나눠주게 되면 Batch의 분포는 [-1, 1]의 범위로 좁혀지게 된다. 즉, 평균과 분산을 구한 후에 정규화시키고, 다시 Scale과 Shift 연산을 위한 감마와 베타가 추가됨으로써, 정규화시켰던 부분을 원래대로 롤백하는 Identity Mapping이 가능하고, 학습을 통해 감마와 베타를 정할 수 있기 때문에 단순 정규화보다 훨씬 뛰어난 학습이 가능해지게 된다.
![image](https://github.com/user-attachments/assets/41700594-5b5d-4a94-89a0-19eb7716cc8c)

  
BatchNormalization은 역전파를 통한 학습이 가능하며, 역전파 시에는 아래와 같은 연쇄 법칙이 적용된다.
![image](https://github.com/user-attachments/assets/73747dc7-7451-4626-9adf-d2159e052ab9)

위 수식을 아래와 같은 코드로 나타내었다.
![image](https://github.com/user-attachments/assets/6835ae82-ddf9-491a-9b59-1d26d3abd6fe)

- Adam optimizer
Adam optimizer의 수식은 다음과 같다.
![image](https://github.com/user-attachments/assets/cc8f5739-0fd4-4e79-894a-c3f27f9cea75)

이를 코드로 표현하였다. (util_func.py)에서 확인 가능
![image](https://github.com/user-attachments/assets/0b02b43b-cde6-4d17-8e52-2e8b63881ebe)

### Neural Net Using PyTorch - init
![image](https://github.com/user-attachments/assets/149ae9a0-0611-4e82-98d1-ce55fde24715)

   numpy로 구현한 Neural Net과 동일하게 구성하였다. init.kaiming._unuform_은 He 초기화이다.

## 학습 과정에 대한 설명
![image](https://github.com/user-attachments/assets/f7e9b606-f72f-46d3-b2f2-7785787b126f)

 update_params는 SGD optimizer에서, update_params_adam은 Adam optimizer에서 사용된다.
Affine계층의 가중치 W에 대하여 학습이 이루어진다. 아래와 같이 기울기(dw)*학습률 값이 마이너스되며 연산된다. adam에서 dw를 구하는 과정은 5절의 adam optimizer 코드를 참고한다.
![image](https://github.com/user-attachments/assets/ec936280-58ad-4dd4-b693-387db0b1d386)

Batch normalization계층의 경우 가중치 감마와 베타에 대해 학습이 이루어진다. 자세한 학습 과정은 5절의 BatchNorm 코드를 참고한다.
 
## 결과 및 분석
### Neural Net Using Numpy Loss(SGD)
![image](https://github.com/user-attachments/assets/d26eb59f-e574-4132-8cb3-e9bb14cdb6f1)

cost 그래프와 accuracy는 아래와 같다.loss값과 관계없이 정확도를 보면 학습이 제대로 이루어짐을 알 수 있 다.loss function의 코드가 잘못된 것이 아닌가 하였는데 디버깅을 해보고 교재에서의 코드를 참고해봐도 문제를 찾 을 수 없었다. batch_normalization는 수식을 보고 코드를 구현하였기 때문에 실사용에서 문제가 생긴 것으로 예상 된다,
   
### Neural Net Using Numpy Loss(Adam)
![image](https://github.com/user-attachments/assets/16bde0bc-040d-40ef-8a15-a0c2bdb3e2d6)

7.1절과 가티 잘못된 cost값이 나오는 것으로 판단된다. accuracy를 보면 학습이 제대로 이루어짐을 확인할 수 있 다. 실제 예측값을 확인해보았을 때도 거의 다 맞추는 것을 확인할 수 있었다. 정확도는 최대 97%정도로 같은 epoch을 돌렸을 때 Adam이 더 좋았다.
   
### Neural Net Using PyTorch (SGD)
  ![image](https://github.com/user-attachments/assets/7b3c84df-f8b5-47d8-8dc7-98ace6449be5)

### Neural Net Using PyTorch (Adam)
![image](https://github.com/user-attachments/assets/10cbe29d-24c1-4773-bd37-2a9f2dd9d4eb)

pytorch를 이용할 때 numpy보다 훨씬 효율이 좋았고 만약 epoch을 더 돌리면 더 좋은 정확도를 보일 것으로 예측 된다
  
