import tensorflow as tf
import numpy as np

x_data = np.array( 
        [[73., 80., 75.],
        [93., 88., 93.],
        [89., 91., 90.],
        [96., 98., 100.],
        [73., 66., 70.]])
y_data =np.array(  
        [[152.],
        [185.],
        [180.],
        [196.],
        [142.]])

print(x_data.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))  
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1.46e-5))

tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)