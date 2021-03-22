import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_datasets as tfds


class Block(tf.keras.Model):

    def __init__(self, filters, kernel_size, repetitions, pool_size = 3, strides = 3):

        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        for i in range(self.repetitions):
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size, strides=strides)

    def call(self, inputs):
        conv2D_0 = self.conv2D_0

        # Connect the conv2D_0 layer to inputs
        x = self.conv2D_0(inputs)
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
            conv2D_i = vars(self)[f'conv2D_{i}']

            # Use the conv2D_i and connect it to the previous layer
            x = conv2D_i(x)
        max_pool = self.max_pool

        return max_pool

class MyAlexNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyAlexNet, self).__init__()
        self.block_1 = Block(filters=64, kernel_size=11, repetitions=1)
        self.block_2 = Block(filters=64, kernel_size=5, repetitions=1)
        self.block_2 = Block(filters=64, kernel_size=3, repetitions=3)

        self.flatten = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier =tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

model = MyAlexNet(10)
def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir='./data')
dataset = dataset.map(preprocess).batch(32)

# train the model.
model.fit(dataset, epochs=1)

model.summary()
