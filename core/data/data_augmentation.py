import torch
from torchvision import transforms as T

IMAGENET_MEAN = torch.Tensor((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
IMAGENER_STD  = torch.Tensor((0.229, 0.224, 0.225)).reshape(-1, 1, 1)

# ######################################################################################################################
#                                                   AUGMENTATION
# ######################################################################################################################
class ImageTransforms:
    def __init__(self, imgSize):
        """
        Class that returns 2 augmented version of an image. A very basic transformation could look something like this.
        Also, do not forget to check the ReadMe.MD for more details.

        Template:
            T.Compose([
                        T.transforms.Resize(size=self.imgSize),
                        [...]
                        T.transforms.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENER_STD)
                    ])

        Example:
            augImage = ImageTransforms((128, 128))
            aug1, aug2 = augImage(image)

        :param imgSize: tuple
        """

        self.imgSize = imgSize
        self.firstAugTrans  = self.get_first_transforms()
        self.secondAugTrans = self.get_second_transforms()

    # ================================================== GET FIRST TRANSFORM ===========================================
    def get_first_transforms(self):
        raise NotImplementedError('Please implement the first transform')       # <======= YOUR TASK #1.1

    # ================================================== GET SECOND TRANSFORM ==========================================
    def get_second_transforms(self):
        raise NotImplementedError('Please implement the second transform')       # <====== YOUR TASK #1.2

    # ================================================== CALL ==========================================================
    def __call__(self, images):
        return self.firstAugTrans(images), self.secondAugTrans(images)


# ================================================== MAIN TEST =========================================================
if __name__ == '__main__':
    image    = torch.rand((512, 512, 3)).numpy()        # dummy image
    augImage = ImageTransforms((128, 128))              # transforms

    aug1, aug2 = augImage(image)
