import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_datasets as tfds
from tensorflow.keras.layers import concatenate
#
class StemBlock(tf.keras.Model):
    def __init__(self, pool_size=3, strides=3):
        super(StemBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu',padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')


    def call(self, input_tensor):

        x = self.conv1(input_tensor)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        max_pool = self.max_pool(x)
        return max_pool


class InceptionBlock(tf.keras.Model):
    def __init__(self, repetitions = 5, pool_size=2, strides=2):
        super(InceptionBlock, self).__init__()
        self.repetitions = repetitions

        # Define a conv2D_0, conv2D_1, etc based on the number of repetitions

        self.conv2D_0 = tf.keras.layers.Conv2D(filters = 64, kernel_size=1, activation='relu', padding='same')
        self.conv2D_1= tf.keras.layers.Conv2D(filters= 64, kernel_size=1, activation='relu', padding='same')
        self.conv2D_2= tf.keras.layers.Conv2D(filters= 64, kernel_size=1, activation='relu', padding='same')
        self.conv2D_3 = tf.keras.layers.Conv2D(filters= 64, kernel_size=5, activation='relu', padding='same')
        self.conv2D_4= tf.keras.layers.Conv2D(filters= 64, kernel_size=3, activation='relu', padding='same')
        self.conv2D_5= tf.keras.layers.Conv2D(filters= 64, kernel_size=1, activation='relu', padding='same')


        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size, strides=strides)

    def call(self, inputs):

        # Connect the conv2D_0 layer to inputs
        conv2D_0 = self.conv2D_0(inputs)
        conv2D_1= self.conv2D_1(inputs)
        conv2D_2= self.conv2D_2(inputs)
        max_pool = self.max_pool(inputs)

        conv2D_3 = self.conv2D_3(conv2D_0)
        conv2D_4= self.conv2D_4(conv2D_1)
        conv2D_5= self.conv2D_5(max_pool)
        concat = concatenate([conv2D_3, conv2D_4, conv2D_5,conv2D_2])
        return concat

class InceptionModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(InceptionModel, self).__init__()
        self.inception_block_1 = InceptionBlock()
        self.inception_block_2 = InceptionBlock()
        self.max_pool_1 = tf.keras.layers.MaxPooling2D(3, 3)

        self.inception_block_3 = InceptionBlock()
        self.inception_block_4 = InceptionBlock()
        self.inception_block_5 = InceptionBlock()
        self.inception_block_6 = InceptionBlock()
        self.max_pool_2 = tf.keras.layers.MaxPooling2D(3, 3)

        self.avg_pool_1 = tf.keras.layers.AvgPool2D(7, 7)
        self.inception_block_7 = InceptionBlock()
        self.inception_block_8 = InceptionBlock()
        self.avg_pool_2 = tf.keras.layers.AvgPool2D(5, 5)
        self.avg_pool_3 = tf.keras.layers.AvgPool2D(5, 5)
        self.inception_block_9 = InceptionBlock()


        self.flatten_1 = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.conv2D_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')
        self.fc_1 = tf.keras.layers.Dense(1024, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier_1 =tf.keras.layers.Dense(num_classes, activation='softmax')

        self.flatten_2 = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.conv2D_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')
        self.fc_2 = tf.keras.layers.Dense(1000, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier_2 =tf.keras.layers.Dense(num_classes, activation='softmax')

        self.flatten_3 = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function

        self.fc_3 = tf.keras.layers.Dense(1000, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier_3 = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, inputs):

        inception_block_1 = self.inception_block_1(inputs)
        inception_block_2 = self.inception_block_2(inception_block_1)
        max_pool_1 = self.max_pool_1(inception_block_2)

        inception_block_3 = self.inception_block_3(max_pool_1)
        inception_block_4 = self.inception_block_4(inception_block_3)
        inception_block_5 = self.inception_block_5(inception_block_4)
        inception_block_6 = self.inception_block_6(inception_block_5)
        inception_block_7 = self.inception_block_7(inception_block_6)

        max_pool_2 = self.max_pool_2(inception_block_7)
        inception_block_8 = self.inception_block_8(max_pool_2)
        inception_block_9 = self.inception_block_9(inception_block_8)
        avg_pool_1 = self.avg_pool_1(inception_block_9)

        flatten_2 = self.flatten_2(avg_pool_1)
        fc2 = self.fc2(flatten_2)
        output_2 = self.classifier_2(fc2)

        return output_2


def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

# create a ResNet instance with 10 output units for MNIST
inception = InceptionModel(10)
inception.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir='./data')
dataset = dataset.map(preprocess).batch(32)

# train the model.
inception.fit(dataset, epochs=1)

inception.summary()
