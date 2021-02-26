import tensorflow as tf
import numpy as np
        
x_data = np.array(# 다리 수, 날개 수
    [[4, 0], #돼지
    [2, 2], #비둘기 
    [2, 0], #성다연
    [2, 2], #독수리
    [4, 0]]) #소
          
y_data = np.array( # 0=포유류, 1=조류
    [[0],
    [1],
    [0],
    [1],
    [0]])

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=500)

print(tf.model.predict(np.array([[2,0]])))