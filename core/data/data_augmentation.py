import torch
from torchvision import transforms as T

IMAGENET_MEAN = torch.Tensor((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
IMAGENER_STD  = torch.Tensor((0.229, 0.224, 0.225)).reshape(-1, 1, 1)

# ######################################################################################################################
#                                                   AUGMENTATION
# ######################################################################################################################
class ImageTransforms:
    def __init__(self, imgSize):
        self.imgSize = imgSize

        self.trainTransforms = self.get_train_transforms()
        self.valTransforms   = self.get_val_transforms()

    def get_train_transforms(self):
        colorJitter = T.ColorJitter(0.8, 0.8, 0.8, 0.3)
        return T.Compose([
                    T.RandomResizedCrop(size=self.imgSize),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([colorJitter], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENER_STD)
                ])

    def get_val_transforms(self):
        return T.Compose([
                    T.transforms.Resize(size=self.imgSize),
                    T.transforms.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENER_STD)
                ])

    def __call__(self, images):
        return self.trainTransforms(images), self.valTransforms(images)