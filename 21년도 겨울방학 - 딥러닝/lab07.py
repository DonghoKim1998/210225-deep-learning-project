# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

x_raw = [[100, 95, 90, 100],
          [90, 100, 80, 90],
          [90, 90, 100, 100],
          [60, 50, 40, 50],          
          [40, 40, 50, 70],
          [50, 50, 40, 60],
          [10, 10, 30, 20],
          [20, 20, 10, 20]]

y_raw = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

x_data = min_max_scaler(x_data)

INPUT_LEN = 4
NB_CLASSES = 3
LEARNING_RATE = 0.1111111111
EPOCH = 100011111

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=INPUT_LEN, units=NB_CLASSES)) 

tf.model.add(tf.keras.layers.Activation('softmax'))

tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=LEARNING_RATE), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=EPOCH)

print('--------------')
a = tf.model.predict(np.array([[0.9, 0.9, 1, 1]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))

b = tf.model.predict(np.array([[0.5, 0.5, 0.6, 0.7]]))
print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))

c = tf.model.predict(np.array([[0.3, 0.4, 0.3, 0.3]]))
c_onehot = tf.model.predict_classes(np.array([[2, 2, 1, 2]]))
print(c, c_onehot)
print('--------------')