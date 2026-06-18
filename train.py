import tensorflow as tf
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

xtrain = xtrain /255
xtest = xtest/255

inputs = tf.keras.layers.Input((32, 32,3))

skip = tf.keras.layers.Conv2D(filters = 64,kernel_size =(1,1) ,padding="same") (inputs)
print(skip.shape)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) ,padding="same") (skip)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),padding="same") (x)
x = tf.keras.layers.BatchNormalization() (x)
print(x.shape)
x = x + skip
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.MaxPooling2D((2, 2)) (x)

#BLOCK 2
skip = x
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) ,padding="same") (skip)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),padding="same") (x)
x = tf.keras.layers.BatchNormalization() (x)
print(x.shape)
x = x + skip
x = tf.keras.layers.Activation("relu") (x)

#BLOCK 3
skip = x
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) ,padding="same") (skip)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),padding="same") (x)
x = tf.keras.layers.BatchNormalization() (x)
print(x.shape)
x = x + skip
x = tf.keras.layers.Activation("relu") (x)

#BLOCK 4
skip = x
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) ,padding="same") (skip)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),padding="same") (x)
x=tf.keras.layers.BatchNormalization() (x)
print(x.shape)
x = x + skip
x=tf.keras.layers.Activation("relu") (x)

#BLOCK 5
skip = x
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3) ,padding="same") (skip)
x = tf.keras.layers.BatchNormalization() (x)
x = tf.keras.layers.Activation("relu") (x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3),padding="same") (x)
x=tf.keras.layers.BatchNormalization() (x)
print(x.shape)
x = x + skip
x=tf.keras.layers.Activation("relu") (x)

x = tf.keras.layers.GlobalAveragePooling2D() (x)
x = tf.keras.layers.Dense(64, "relu") (x)

output = tf.keras.layers.Dense(10,activation="softmax") (x)
model = tf.keras.Model(inputs, output)
model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam",metrics= ["accuracy"])

model.fit(xtrain, ytrain, epochs=5, validation_split=0.2)
file = model.save("cifar10-residual-cnn.keras")

