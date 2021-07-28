# Image Classification Using Deep Learning

**Learning and Building Convolutional Neural Networks using PyTorch. Models, selected are based on number of citation of the paper with the help of [paperwithcode](https://paperswithcode.com/) along with unique idea deviating from typical architecture like using transformers for CNN.**

![Image Classification Using Deep Learning](Images/cnn.jpg)

### Content

<details>
  <summary>Models include CNN architecture, MLP based classifiers, transformer based model and many more to come.</summary>
<br>
<details>
  <summary>🔥 AlexNet</summary>
<p>

    Before AlexNet, SIFT(scale-invariant feature transform), SURF or HOG were some of the hand tuned 
    feature extractors for Computer Vision.

    In AlexNet, Interestingly in the lowest layers of the network, the model learned feature extractors 
    that resembled some traditional filters. Higher layers in the network might build upon these 
    representations to represent larger structures, like eyes, noses, blades of grass, and so on. Even 
    higher layers might represent whole objects like people, airplanes, dogs, or frisbees. Ultimately, 
    the final hidden state learns a compact representation of the image that summarizes its contents such 
    that data belonging to different categories can be easily separated.
    
    Challenges perceived before AlexNet:

    Computational Power:

    Due to the limited memory in early GPUs, the original AlexNet used a dual data stream design, so that 
    each of their two GPUs could be responsible for storing and computing only its half of the model. 
    Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models across GPUs 
    these days.

    Data Availability:

    ImageNet was released during this period by researchers under Fei-Fei Li with 1 million images, 1000 images 
    per class with total of 1000 class.

    Note:
    Instead of using ImageNet, I am using MNIST and resizing the image to 224 x 224 dimension to make it justify 
    with the AlexNet architecture.
</p>
  
 <img src="Images/alexnet.png" alt="AlexNet">
</details>

<details>
  <summary>🔥 VGGNet</summary>
  <p>
    
    VGGNet brings in the idea of buliding a block of network like a template unlike previous CNN architecture 
    where the network is built layer by layer with increasing complexity.
  
    VGG network helps researchers think in terms of block of network. A typical network of convolution would 
    require following steps
  
    * Conv with padding for maintaining resolution.
    * Activation Function
    * Pooling for spatial downsampling
  
    Note: I don't recommend running this until you have GPU, the number of parameters is increased by huge number 
    compared to AlexNet.

    Changes made for faster convergence and which deviates from VGG Net is learning rate is changed to 0.05 and 
    reduce the number channels by 1/4th.

    Check out the loss with these changes, since lr is high compared to typical values, the loss moves drastically 
    and then converges. Without Xavier's Weight Initialization, the model performs poorly.

    Why VGG is slower than AlexNet?

    One reason is that AlexNet uses (11x11 with a stride of 4), while VGG uses very small receptive fields (3x3 
    with a stride of 1) which makes it slower to move over the image and overall the parameters are 3 times the 
    AlexNet.
    
  </p>
<img src="Images/vggnet.png" alt="VGGNet">
</details>
<details>
  <summary>🔥 NIN</summary>
<p>
  
    Network In Network introduced one of the key concept in deep neural network of dimension downsampling/upsampling 
    using 1x1Conv layer. It applies MLP on the channels for each pixel separately.

    The idea behind NiN is to apply a fully-connected layer at each pixel location (for each height and width). 
    If we tie the weights across each spatial location, we could think of this as a 1×1 convolutional layer 
    or as a fully-connected layer acting independently on each pixel location. Another way to view this is to think
    of each element in the spatial dimension (height and width) as equivalent to an example and a channel as equivalent
    to a feature.

    NIN introduces the 1x1 Convolution. Smaller batch size results in better performance even though it is slow.
  
  </p>
  
<img src="Images/nin.png" alt="NIN">
</details>
<details>
<summary>🔥 GoogLeNet</summary>
<p>
  
    It combined ideas from NIN and VGG network introducing InceptionV1 also known as GoogLeNet. 

    In AlexNet, we've used 11x11 Conv, in NIN, we used 1x1 Conv. And in this paper, we identify
    among different kernel, which sized convolutional kernels are best. It is the version 1 of Inception
    model. 

    GoogLeNet introduces the concept of parallel concatenation of networks. We bulid Inception block and 
    which is repeated in the architecture.

    Some intution on the architecture, since the various different sized filters are at work, different spatial
    relations are extracted by different filters efficiently. It also allocates different amt of parameters
    across different filters.

    * 1×1 convolutions reduce channel dimensionality on a per-pixel level. Maximum pooling reduces the resolution.
    * If you're wondering how these dimensions were decided, it is based on trial and error & based on ImageNet 
    Dataset.
</p>
<img src="Images/googlenet.png" alt="GoogLeNet">
</details>
<details>
<summary>🔥 BatchNorm + ConvNet</summary>
<p>
    BatchNorm was introduced as a concept to **normalize the mini-batches traversing through the layer** and had an 
    impactful results having **regularization** effect. But why BatchNorm is effective is quite unclear? the author 
    suggests that BatchNorm reduce internal variant shift but other researchers  pointed out that the effects which 
    batchNorm is effective against is not related to covariant shift. It is still widely discussed topic in DL.
  
</p>
<img src="Images/batchnorm.png" alt="BatchNorm + ConvNet">
</details>
<details>
<summary>🔥 ResNet</summary>
<p>
  
    ResNet Architecture has huge influence in current DNN architectures. It introduces the idea of **skip connection**, 
    a concept of **adding** an unfiltered input to the conv layers.

    Why ResNet?

    To understand the network as we add more layers, does it becomes more expressive of the
    task in hand or otherwise.

    Key idea of ResNet is adding more layers which acts as a Identity function, i.e. if our
    underlying mapping function which the network is trying to learn is F(x) = x, then instead
    of trying to learn F(x) with Conv layers between them, we can directly add an skip connection
    to tend the weight and biases of F(x) to zero. This is part of the explanation from D2L.
    Adding new layer led to ResNet Block in the ResNet Architecture.

    In ResNet block, in addition to typical Conv layers the authors introduce a parallel identity 
    mapping skipping the conv layers to directly connect the input with output of conv layers.
    A such connection is termed as Skip Connection or Residual connection.

    Things to note while adding the skip connection to output conv block is the dimensions.Important
    to note, as mentioned earlier in NIN network, we can use 1x1 Conv to increase and decrease the 
    dimension.

    In the code block, we have built ResNet18 architecture:

    There are 4 convolutional layers in each module (excluding the 1×1 convolutional layer). 
    Together with the first 7×7 convolutional layer and the final fully-connected layer, there are 
    18 layers in total. Therefore, this model is commonly known as ResNet-18.
</p>
<img src="Images/resnet.png" alt="ResNet">
</details>
<details>
<summary>🔥 DenseNet</summary>
  <p>
    
    Building upon ResNet, DenseNet introduced the idea of **concatenating** the previous layers 
    output and as well the inputs to the next layers.

    In ResNet, we see how the skip connection added as identity function from the inputs
    to interact with the Conv layers. But in DenseNet, we see instead of adding skip 
    connection to Conv layers, we can append or concat the output of identity function
    with output of Conv layers.

    In ResNet, it is little tedious to make the dimensions to match for adding the skip
    connection and Conv Layers, but it is much simpler in DenseNet, as we concat the 
    both the X and Conv's output.

    The key idea or the reason its called DenseNet is because the next layers not only get
    the input from previous layer but also preceeding layers before the previous layer. So 
    the next layer becomes dense as it loaded with output from previous layers.

    Check Figure 7.7.2 from https://d2l.ai/chapter_convolutional-modern/densenet.html for 
    why DenseNet is Dense?

    Two blocks comprise DenseNet, one is DenseBlock for concat operation and other is 
    transition layer for controlling channels meaning dimensions (recall 1x1 Conv).
  </p>
<img src="Images/Densenet.png" alt="DenseNet">
</details>
<details>
<summary>🔥 Squeeze And Excitation Network</summary>
<p>
  
    A typical convolution network has kernels running through image channels and combining
    the feature maps generated per channel. For each channel, we'll have separate kernel which
    learns the weights through backpropagation.

    The idea is to understand the interdependencies between channels of the images by explicitly
    modeling on it and hence to make the network sensitive to informative features which is further
    exploited in the next set of transformation.

      * Squeeze(Global Information Embedding) operation converts feature maps into single value per channel.
      * Excitation(Adaptive Recalibration) operation converts this single value into per-channel weight.

    Squeeze turns (C x H x W) into (C x 1 x 1) using Global Average Pooling.

    Excitation turns (C x 1 x 1) into (C x H x W) channel weights using 2 FC layer with activation function
    inbetween, then which is expanded as same size as input.

    Rescale the output from excitation operation into feature maps as earlier.

    Based on the depth of the network, the role played by SE operation is differs. At early layers,
    it excites shared low level representation irrespective of the classes. But in later stage, SE 
    network responds differently based input class.
    
    SE Block is simple and is added with existing CNN architecture to enhance the performance like 
    ResNet or Inception V1 etc.

      Reference: https://amaarora.github.io/2020/07/24/SeNet.html
</p>
<img src="Images/senet.png" alt="SENet">
</details>
<details>
<summary>🔥 MLP-Mixer</summary>
  <p>
    
    This particular network doesn't come under convolutional networks as the key idea is to use simple MLP 
    architecture.

    MLP-Mixer is a multi-layer perceptron based model, it uses common techniques like non-linearites, matrix 
    multiplication, normalization, skip connections etc. This paper is very interesting to the fact that when 
    MLP was introduced, it was particular made upfront that the MLP architectures cannot capture translation 
    invariance in an image. 

    Let's see how things have changed. The Network uses a block of MLP Block with two linear layers and one 
    activation function GELU unit. Along with MLPBlock, there are two simple small block called as token mixer 
    and channel mixer. Normalization is done throughout the network using Layer Normalization.

    * First, the image is converted into patches
    * These patches are also called as tokens.
    * MLP is a Feedforward network.
    * In Token Mixer, we mix these tokens using MLP, it learns spatial locations.
    * In Channel Mixer, we mix the channels using MLP, it learns channel dependencies.
    * The we combine of channel mixer and token mixer.
    * It passed into Global Average Pooling and then into Fully connected layer.

    Best tutorial to learn about einops: https://github.com/arogozhnikov/einops/blob/master/docs
  </p>
<img src="Images/mlpmixer.png" alt="MLP-Mixer">
</details>
<details>
<summary>🔥 MobileNet</summary>
<p>
  
    A convolutional neural network with large number of layers is expensive, both interms of memory and the 
    hardware requirement for inference and thus deploying such models in mobile devices is not feasible.

    To overcome the above challenge, a group of researchers from Google built a neural network model 
    optimized for mobile devices referred as MobileNet. Underlying idea of mobilenet is depthwise
    seperable convolutions consisting of depthwise and a pointwise convolution to build lighter models.

    MobileNet introduces two hyperparameters

    * Width Multiplier

    Width muliplier (denoted by α) is a global hyperparameter that is used to construct smaller and less 
    computionally expensive models.Its value lies between 0 and 1.For a given layer and value of α, the 
    number of input channels 'M' becomes α * M and the number of output channels 'N' becomes α * N hence 
    reducing the cost of computation and size of the model at the cost of performance.The computation cost 
    and number of parameters decrease roughly by a factor of α2.Some commonly used values of α are 1,0.75,
    0.5,0.25.

    * Resolution Multiplier

    The second parameter introduced in MobileNets is called resolution multiplier and is denoted by ρ.This 
    hyperparameter is used to decrease the resolution of the input image and this subsequently reduces the 
    input to every layer by the same factor. For a given value of ρ the resolution of the input image becomes 
    224 * ρ. This reduces the computational cost by a factor of ρ2.

    The above parameters helps in trade-off between latency (speed of inference) and accuracy.

    MobileNet is 28 layers neural net represented by both the depthwise convolution and pointwise convolution.

     - Depthwise convolution is the channel-wise n×n spatial convolution. 
     Suppose in the figure above, we have 5 channels, then we will have 5 n×n spatial convolution.

     - Pointwise convolution actually is the 1×1 convolution to change the dimension.
</p>
<img src="Images/mobilenetv1.png" alt="MobileNetV1">
</details>
<details>
  <summary>🔥 InceptionV3</summary>
  <p>
      
    The Inception deep convolutional architecture was introduced as GoogLeNet, here named Inception-v1. 
    Later the Inception architecture was refined in various ways, first by the introduction of batch 
    normalization (Inception-v2). Later by additional factorization ideas in the third iteration 
    which is referred as Inception-v3.

    Factorizing Convolution: Idea is to decrease the number of connections/parameters without reducing
    the performance.

    * Factorizing large kernel into two similar smaller kernels
        - Using 1 5x5 kernel, number of parameters is 5x5=25
        - Using 2 3x3 kernel instead of one 5x5, gives 3x3 + 3x3 = 18 parameters.
        - Number of parameter is reduced by 28%.

    * Factorizing large kernel into two assimilar smaller kernels
        - By using 3×3 filter, number of parameters = 3×3=9
        - By using 3×1 and 1×3 filters, number of parameters = 3×1+1×3=6
        - Number of parameters is reduced by 33%

    * If we look into InceptionV1 i.e. GoogLeNet, we have inception block which uses 5x5 kernel and 3x3 
    kernel, factorizing technique can reduce the number of parameters in the networks.

    Other Changes:

    From InceptionV1, we bring in Auxillary classifier which acts as regularizer. We also see, efficient
    grid size reduction using factorization instead of standard pooling which is expensive and greedy operation.
    Label smoothing, to prevent a particular label from dominating all other class.
</p>
<img src="Images/inceptionv3.png" alt="InceptionV3">
</details>
<details>
  <summary>🔥 Xception</summary>
  <p>
    
    The network uses a modified version of Depthwise Seperable Convolution. It combines
    ideas from MobileNetV1 like depthwise seperable conv and from InceptionV3, the order 
    of the layers like conv1x1 and then spatial kernels.

    In modified Depthwise Seperable Convolution network, the order of operation is changed
    by keeping Conv1x1 and then the spatial convolutional kernel. And the other difference
    is the absence of Non-Linear activation function. And with inclusion of residual 
    connections impacts the performs of Xception widely.
  </p>
  <img src="Images/Xception.png" alt="Xception">
  </details>
<details>
  <summary>🔥 ResNeXt</summary>
  <p>

    ResNeXt is a simple, highly modularized network architecture for image classification. The
    network is constructed by repeating a building block that aggregates a set of transformations 
    with the same topology. The simple design results in a homogeneous, multi-branch architecture 
    that has only a few hyper-parameters to set. This strategy exposes a new dimension, which is 
    referred as “cardinality” (the size of the set of transformations), as an essential factor in 
    addition to the dimensions of depth and width.

    We can think of cardinality as the set of separate conv block representing same complexity as 
    when those blocks are combined together to make a single block.

<img src="https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac" alt='RexNeXt Blog'>

</p>  
  <img src="Images/resnext.png" alt="ResNeXt">
  
</details>
<details>
  <summary>🔥 ViT</summary>
  <p>

    Vision Transformer aka ViT
    
    Transformers are the backbone architecture for many of the NLP architectures like BERT etc. Though, it
    started with focus on NLP tasks, the transformer is used in computer vision space. In Transformer, there
    is an encoder and a decoder block. In ViT, we use transformer's encoder block and use MLP head for 
    prediction.

    We'll discuss about Transformer architecture separately except the notion on data, we'll see how the 
    image is processed in transformer, which was primarily built for sentence tokens. There are series of 
    steps followed to convert image into sequence of token and passed into transformer encoder with MLP.

    * Convert Image into Patches of fixed size.
    * Flatten those patches into sequence of embedding
    * Add positional embeddings
    * Feed the sequence into transformer encoder
    * And predict using MLP block at last.

    I've omitted few notions from transformer architecture like residual connections, multi-head attention
    etc. Each of these concept requires separate blog post.

    Note: ViT was trained on large image dataset with 14M images, and the pretrained model is fine tuned to 
    work with our custom dataset.
    
<p><strong>I strongly recommend going through code blocks, where I've mentioned the flow of an Image through
ViT architecture with all dimensional changes.<strong></p>
  
  </p>
  <img src="https://media.giphy.com/media/ATsWtUsuuFRfq8OhZ7/giphy.gif"; width=80%; alt="ViT">
  
  [Citation Details](https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch")
</details>
<details>
  <summary>🚀 MobileNetV2</summary>
</details>
<details>
  <summary>🚀 Darknet-53</summary>
</details>
<details>
  <summary>🚀 SqueezeNet</summary>
</details>
<details>
  <summary>🚀 ShuffleNet</summary>
</details>
<details>
  <summary>🚀 EfficientNet</summary>
</details>
<details>
  <summary>🚀 CycleMLP</summary>
</details>

🔥 - Up and Running

🚀 - Inprogress  

</details>

<!--##########################################################################################-->

### Create Environment
```python
python -m venv CNNs 
source CNNs/bin/activate 
git clone https://github.com/Mayurji/CNNs-PyTorch.git
```

### Installation
```python
pip install -r requirements.txt
```

### Run
```python
python main.py --model=resnet
```

**Note:** Parameters can be changed in YAML file. The module supports only two datasets, MNIST and CIFAR-10, but you can modify the dataset file and include any other datasets.
