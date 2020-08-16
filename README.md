# Convolutional-Neural-Networks

<p> In deep-learning, a Convolutional Neural Network(CNN) is a class of deep neural networks which is commonly applied to analyzing visual imagery. CNN use relatively little preprocessing compared to other image classification algorithms.</p>

### Architecture of CNN

<p>A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of a series of convolutional layers that convolve with a multiplication or other dot product. The activation function is commonly a RELU layer, and is subsequently followed by additional convolutions such as pooling layers, fully connected layers and normalization layers, referred to as hidden layers because their inputs and outputs are masked by the activation function and final convolution.</p>

#### Convolutional Layer

<p>When programming a CNN, the input is a tensor with shape (number of images) x (image height) x (image width) x (image depth). Then after passing through a convolutional layer, the image becomes abstracted to a feature map, with shape (number of images) x (feature map height) x (feature map width) x (feature map channels). A convolutional layer within a neural network should have the following attributes:

  * Convolutional kernels defined by a width and height (hyper-parameters).
  * The number of input channels and output channels (hyper-parameter).
  * The depth of the Convolution filter (the input channels) must be equal to the number channels (depth) of the input feature map.
  
Convolutional layers convolve the input and pass its result to the next layer. This is similar to the response of a neuron in the visual cortex to a specific stimulus.Each convolutional neuron processes data only for its receptive field. Although fully connected feedforward neural networks can be used to learn features as well as classify data, it is not practical to apply this architecture to images. A very high number of neurons would be necessary, even in a shallow (opposite of deep) architecture, due to the very large input sizes associated with images, where each pixel is a relevant variable. For instance, a fully connected layer for a (small) image of size 100 x 100 has 10,000 weights for each neuron in the second layer. The convolution operation brings a solution to this problem as it reduces the number of free parameters, allowing the network to be deeper with fewer parameters.For instance, regardless of image size, tiling regions of size 5 x 5, each with the same shared weights, requires only 25 learnable parameters. By using regularized weights over fewer parameters, the vanishing gradient and exploding gradient problems seen during backpropagation in traditional neural networks are avoided.</p>

#### Pooling Layer

<p>Convolutional networks may include local or global pooling layers to streamline the underlying computation. Pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. Local pooling combines small clusters, typically 2 x 2. Global pooling acts on all the neurons of the convolutional layer.In addition, pooling may compute a max or an average. Max pooling uses the maximum value from each of a cluster of neurons at the prior layer.Average pooling uses the average value from each of a cluster of neurons at the prior layer.</p>

#### Fully Connected Layer

<p>Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network (MLP). The flattened matrix goes through a fully connected layer to classify the images.</p>

#### Note: *Using a convolutional neural network reduces the pixels(height and width) of the images but increases the number of color channels. This will significantly lowers down the input size of image(especially for very large images like 360X360, 720X720 and so on) for neural network. The main advantages of convolutional layers over just using fully connected layers  are Parameter Sharing and Sparsity of Connections. By parameter sharing, the numbers of parameters are reduced which helps neural network to train more faster and efficiently for large input sizes.Sparsity of connections is that the output value depends only on a small number of inputs in each layer. One layer in CNN is the combination of convolution layer(conv2d) and max-pooling layer.*

## Residual Networks(ResNets)

<p> In practice, having a plain network(not a ResNets) that is very deep makes the optimization algorithm to consume much harder time during training and the training error gets worse if we pick a very deep network. That is why, we will use Residual Network(ResNet) so that we can train very very deep network(even over 100 layers). Even with such amount of layers in ResNet, the training error kind of keep going down which is a good sign.</p>
<p> ResNets uses *skip connection* technique which allows to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network. This network is built out of something called Residual Blocks and stacked together to form a deeper network.</p>

## Inception Network

<p> The main idea of inception network is we can use different filter size ie 1X1, 3X3, 5X5 and Pooling layer at the same time in a CNN and concatenate all the outputs and let the network learn whatever the combinations of these filter sizes it wants. The 1X1 filter layer helps to reduce the number of channels and keep the size of image same.</p>
<p>If we are building a layer of a neural network and we don't have to decide which size of filter layer to choose, the inception module lets us to do them all and concatenate the result.But using this module will have computation cost problem. To solve this, the 1X1 layer is used as a bottleneck layer to compute other layer(3X3,5X5 and Pooling).</p>

#### Note: *One of the best way to get intuition about best working CNN is to see some case studies or research papers. Some of the effective Classcical Neural Networks are LeNet-5, AlexNet and VGG. Studying these networks can be very helpful to get better intuition about these networks.*

