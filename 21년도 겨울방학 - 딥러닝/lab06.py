# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

x_raw = [[10, 9, 9, 10],
          [9, 10, 8, 9],
          [9, 9, 10, 10],
          [6, 5, 4, 5],          
          [4, 4, 5, 7],
          [5, 5, 4, 6],
          [1, 1, 3, 2],
          [2, 2, 1, 2]]

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

INPUT_LEN = 4
NB_CLASSES = 3
LEARNING_RATE = 0.1
EPOCH = 100

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=INPUT_LEN, units=NB_CLASSES)) 

tf.model.add(tf.keras.layers.Activation('softmax'))

tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=LEARNING_RATE), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=EPOCH)

print('--------------')
a = tf.model.predict(np.array([[9, 9, 10, 10]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))

b = tf.model.predict(np.array([[4, 4, 5, 7]]))
print(b, tf.keras.backend.eval(tf.argmax(b, axis=1)))

c = tf.model.predict(np.array([[2, 2, 1, 2]]))
c_onehot = tf.model.predict_classes(np.array([[2, 2, 1, 2]]))
print(c, c_onehot)
print('--------------')