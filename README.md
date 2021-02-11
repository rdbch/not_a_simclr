# SimCLR - A naive tutorial

## Installation
Conda
```

```

pip
```
$ pip install -r requirements.txt
```

## Theoretical informations

SimCLR is a framework of contrastive learning that was introduced by Ting Chen et. al. in [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709v3.pdf) at ICML 2020(😮), that allows one to learn good representations form data without any express supervision. What does this mean? Think of fitting a CNN model on a completly new dataset, instead of training it from scratch, it is a very common practice to start with the weights trained of a learge and generalistic dataset such as ImageNet (1000 classes with 1000 images/class). This speeds up the training process and helps one achieve better results, because the used encoder, learned very good representations from data. However, when we don't have access to such a model, or when we want to train our model on a new dataset that has very few labeled samples we can use this method to obtain a similar effect.  In the image below, one can see that by using this method, one can acieve performances similar with supervised approaches. 

<img src="C:\Users\W9KHSOK\AppData\Roaming\Typora\typora-user-images\image-20210211084309477.png" alt="image-20210211084309477" style="zoom: 50%;" />



Long story short, this SimCLR is a 🏃‍♂️training method🏃‍♂️ that can be used to create a pretrained model for your custom dataset and not requiring any labels. It does thisby maximizing the agreement between differently augmented views of the same image via a *contrastive loss* in the latent space. The produced network, can be further used to solve tasks, but keep it will require some sorth of supervision to do so. 

 During training, this framework is composed from 4 main components (for each section, more information will be presented in the **Task** section of each module): 

<img src="C:\Users\W9KHSOK\Pictures\image-20210211084754141.png" alt="image-20210211084754141" style="zoom: 67%;" />



1. **Image augmentation :** Module responsible with generating two correlated views of the same example. 
2. **Encoder:** An image incoder (CNN) that is used to extract latent representation vectors from the augmented samples . 
3. **Projection Head:** A small neural network, a few linear units with a few hidden layers, that is used to map the latent space of the encoder output to another latent space where contrastive loss can be applied. 
4. **Contrastive Loss Function:** Given a set of examples including a positive(simialr) pair of examples, the contrastive prediction task has to identify them.

## Tutorial

For this tutorial, a started project was developed such that, it will allow one to only focues on important things. Besides this, if everything os done correctly, one will be able to see the latent space representation in Tensorboard. Because, we do not have access to a vaste amount of computing resources, we will use the CIFAR10 dataset. 

### Data augmentation

In this first segction, one will have to implement the image augmentation module. The module should be a callable object that takes as parameter an image and returns 2 augmented versions of it. For more information of what transformations are used, check page 12 of the the SimCLR paper ( [arxiv](https://arxiv.org/pdf/2002.05709v3.pdf) ). A base code structer for this class is provided and one will modify the following file:

``` 
core/data/data_augmentation.py
```



**Sample usage**

```python
augModule = ImageTransforms(imgSize)
image     = cv2.imread(...) # H, W, C

augImage1, augImage2 = augModule(image)
```





**Pseudo-python-code hint**

```python
class ImageTransforms:
    def __init__(self, imgSize):
        self.firstAugTrans  = sefl.get_first_aug(imgSize)
        self.secondAugTrans = sefl.get_second_aug(imgSize)
   # . . .      
   # define get_first_aug & get_second_aug
   # . . .
        
    def __class__(self, image)
    	firstAug  = self.firstAugTrans(image)
        secondAug = self.secondAugTrans(image)
      
        return firstAug, secondAug
```



### Neural network



## References
