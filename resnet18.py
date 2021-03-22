import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_datasets as tfds


class IdentityBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ConvBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()
        self.conv2 = tf.keras.layers.Conv2D(filters,kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, input_tensor])

        return x


class ResNet(tf.keras.Model):
    def __init__(self, num_classes, repetitions = 3):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pooling = tf.keras.layers.MaxPooling2D((3, 3))
        self.id_block_1 = IdentityBlock(64, 3)
        self.id_block_2 = IdentityBlock(64, 3)
        self.global_ppol = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.repetitions = repetitions

        for i in range(self.repetitions):
            # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
            vars(self)[f'conv_block_{i}'] = ConvBlock(64, 3)
            vars(self)[f'conv_id_block_{i}'] = IdentityBlock(64, 3)
    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pooling(x)
        x = self.id_block_1(x)
        x = self.id_block_2(x)

        conv_id_block = self.conv_id_block_0
        conv_block_0 = self.conv_block_0

        x = self.conv_id_block_0(x)
        x = self.conv_block_0(x)
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
            conv_id_block_i = vars(self)[f'conv_id_block_{i}']
            conv_block_i = vars(self)[f'conv_block_{i}']
            # Use the conv2D_i and connect it to the previous layer
            x = conv_id_block_i(x)
            x = conv_block_i(x)

        x = self.global_ppol(x)

        return self.classifier(x)



def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

# create a ResNet instance with 10 output units for MNIST
resnet = ResNet(10)
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir='./data')
dataset = dataset.map(preprocess).batch(32)

# train the model.
resnet.fit(dataset, epochs=1)

resnet.summary()
