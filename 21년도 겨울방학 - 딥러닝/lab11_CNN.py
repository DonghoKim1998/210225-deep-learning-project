import glob
import numpy as np
import tensorflow as tf
import random
import matplotlib.pylab as plt
from PIL import Image

mnist = tf.keras.datasets.mnist
# keras에서 제공하는 손글씨 mnist를 사용

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# mnist에서 데이터 load
# x는 image 정보가 담겨있고
# y는 그 이미지의 숫자가 담겨 있다.

my_image_list = glob.glob('my data/*.png')
# glob을 통해서 my data 폴더 밑의 모든 png 파일의 이름을 불러옴

x_test = []
y_test = []

for i in my_image_list:
    label = int(i[i.index('\\')+1:i.index('.')])
    # 파일명 i 가 'my data\0.png' 식으로 저장되어 있어서 여기서 숫자 0만 떼어내어 label로 사용
    img = Image.open(i)
    img = img.convert('L')
    # 파일명 i 로 이미지 파일을 불러온 후, greyscale 이미지로 변환

    y_test.append(label) # label은 y_test에 추가
    x_test.append(np.array(img)) # image는 numpy array로 바꾸어 x_test에 추가

x_test= np.array(x_test)
y_test= np.array(y_test)

# matplot library 를 imshow 기능을 통해 이미지를 불러와 직접 확인
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_test[0])
plt.show()
# train 데이터와 test 데이터를 하나 씩 확인

print(len(x_test))
print(len(y_test))

x_test = 255 - x_test # imshow를 통해 확인했을때, 두 이미지가 서로 색이 반전되어 있었음. 그것을 맞추어 줌
x_test = x_test / 255
x_train = x_train / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# one hot encode y data
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# hyper parameters
learning_rate = 1.46e-3
training_epochs = 12
batch_size = 128

tf.model = tf.keras.Sequential()
# L1
tf.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# L2
tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# L3 fully connected
tf.model.add(tf.keras.layers.Flatten())
tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))

tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.summary()

tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

y_predicted = tf.model.predict(x_test)
for i in range(len(y_test)):
    print("real:",np.argmax(y_test[i]), end=', ')
    print("predicted:", np.argmax(y_predicted[i]))
