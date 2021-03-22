# Building Neural Network architectures from scratch with TensorFlow
I have built simple versions of some Neural Network architectures (Alexnet, Inception-v1, Resnet-18, Vgg-16) from scratch and trained them with Distribution Strategies by using TensorFlow.

## 1. AlexNet (2012)

With 60M parameters, AlexNet has 8 layers — 5 convolutional and 3 fully-connected. AlexNet just stacked a few more layers onto LeNet-5. At the point of publication, the authors pointed out that their architecture was “one of the largest convolutional neural networks to date on the subsets of ImageNet.”

![alt text](https://miro.medium.com/max/2000/1*2DT1bjmvC-U-lrL7tpj6wg.png)


## 2. VGG-16 (2014)

VGG-16 has 13 convolutional and 3 fully-connected layers, carrying with them the ReLU tradition from AlexNet. This network stacks more layers onto AlexNet, and use smaller size filters (2×2 and 3×3). It consists of 138M parameters and takes up about 500MB of storage space.

![alt text](https://miro.medium.com/max/2000/1*_vGloND6yyxFeFH5UyCDVg.png)


## 3. Inception-v1 (2014)


This architecture with 5M parameters is called the Inception-v1. Here, the Network In Network (see Appendix) approach is heavily used, as mentioned in the paper. This is done by means of ‘Inception modules’.

![alt text](https://miro.medium.com/max/4800/1*KnTe9YGNopUMiRjlEr3b8w.png)

## 4.ResNet-18


From the past few CNNs, we have seen nothing but an increasing number of layers in the design, and achieving better performance. But “with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly.”

![alt text](https://www.researchgate.net/profile/Muhammad-Hasan-27/publication/323063171/figure/fig1/AS:603178554904576@1520820382219/Proposed-Modified-ResNet-18-architecture-for-Bangla-HCR-In-the-diagram-conv-stands-for.png)




