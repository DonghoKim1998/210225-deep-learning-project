import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

xy_data = np.loadtxt('./datasets/ppg_bp.csv', delimiter=',', dtype=np.float32)
x_data = xy_data[:,2:]
y_data = xy_data[:,:2]
print("x_data shape : ", x_data.shape)
print("y_data shape : ", y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

tf.model = tf.keras.Sequential()
#tf.model.add(tf.keras.layers.Dense(units=1, input_dim=200, activation='linear'))
tf.model.add(tf.keras.layers.Dense(units=2000, input_dim=200, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=1000, input_dim=100, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=1000, input_dim=100, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=1000, input_dim=100, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=100, activation='relu'))

tf.model.summary()

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-4))
history = tf.model.fit(x_train, y_train, epochs=300)

pd = tf.model.predict(x_test)
print(mse(y_test, pd))
# = print(np.mean((y_test - pd)**2))



"""

"""
