import tensorflow as tf

x_train = [1, 2, 3, 4] # 입력값들 (x)
y_train = [3, 5, 7, 9] # 출력값들 (y)
# 1 이 입력되면 출력은 3, 2 가 입력되면 출력은 5...
# 출력 (y)는 입력 (x)에 대하여 y = 2x + 1 의 관계를 가진다.
# Linear Regression을 통하여 이를 유추할 것.

tf.model = tf.keras.Sequential()
# 모델을 생성한다. Seqential(순차)모델은 기본 모델이라고만 알고 있자.

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# tf.moel.add를 통하여 모델에 레이어를 추가할 수 있다.
# tf.keras.layers.Dense(units=1, input_dim=1)는 입력과 출력이 하나인 기본 레이어.
# Hypothesis H(x) = Wx + b 가 내부적으로 구현되어 있다. 


sgd_optimizer = tf.keras.optimizers.SGD(lr=0.1) 
# optimizer를 생성한다. 여기서는 Stochastic Gradient Descent (확률적 경사 하강법) 사용함.
# learning rate는 0.1로 설정

tf.model.compile(loss='mse', optimizer=sgd_optimizer) 
# loss 는 cost와 같은 의미. 여기서는 MSE(Mean Square Error)를 사용하였다.
# MSE는 제곱평균오차로, 그냥 평균오차를 사용하면 오차가 -10, 10 일 경우 평균 0으로 오차가 없는 것이 된다.
# 따라서, 평균오차를 제곱하여 음수가 나오지 않게 하여 사용한다.

tf.model.summary()
# 모델에 대한 요약 정보를 요청

tf.model.fit(x_train, y_train, epochs=100)
# 순서대로
# x_train을 입력으로 사용하고
# y_train을 출력하는 것을 목표로
# 100번 반복해서 학습 (epochs는 세대 수. 즉, 반복수라고 생각하면 됨.) 

x_test = [5, 6, 7, 8] 
# test 데이터. 인공지능이 y= 2x+1 이라는 규칙을
# 잘 학습했다면 출력은 11, 13, 15, 17이 나와야함. 

y_predict = tf.model.predict(x_test)
print(y_predict)
# model을 사용해 x_test에 대하여 예측하는 함수
# print(y_predict) 하여 예측결과를 확인한다.