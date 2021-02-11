from torch import nn

from core.nn.blocks             import ResBlock, Conv2dBlock
from core.nn.utils.configurable import Configurable


########################################################################################################################
#                                               RESNET
########################################################################################################################
class ResNet(nn.Module, Configurable):
    def __init__(self, inChNo=3, outChNo = None, archType = '18', **kwargs):
        """
        Class that servers as a common interface for ResNet18 and ResNet34 from
        "Deep Residual Learning for Image Recognition by Kaiming el. al."
        For more details see:  https://arxiv.org/pdf/1512.03385.pdf

        Because the implementation provided by PyTorch does not provide enough customization options just by using
        the parameters, I decided to do this implementation. Having the same configuration, the model weights can be
        loaded from PyTorch.

        Example:
            resnet = ResNet( inChNo           = 3,                  outChNo         = 512,
                             norm             = 'BatchNorm2d',      normKwargs      = {},
                             activ            = 'ReLU',             activKwargs     = {},
                             lastActiv        =  None,              lastActivKwargs = {},
                             strides          = [2, 2, 2, 2, 2, 2], downType        = 'maxpool',
                             lastPool         = True,               archType        = '18',
                             firstLayerKernel = 7,                  lastLayerKernel = 1).build()
            latent = resnet(image)

        :param inChNo           : Number of input channels
        :param outChNo          : Number of output channels (if None or 0, not last layer is added)
        :param archType         : The architecture (default: '18') ['18', '34']
        :param activ            : Activation used on the output of each layer (default: ReLU)*
        :param activKwargs      : Kwargs for activ (default: {'inplace' : True})
        :param lastActiv        : Activation applied on the last layer (defaul: None) *
        :param lastActivKwargs  : Kwargs for the last activ (default: {})
        :param norm             : Normalization layer name * (default: BatchNorm2d)
        :param normKwargs       : Kwargs passed to normalization layer (default: {})
        :param firstLayerKernel : Size of the first kernel (default: 7)
        :param lastLayerKernel  : Size of the last kernel (default: 1)
        :param kernel           : Size of the usual convolution kernel (default: 3)
        :param poolKernel       : Size of the pooling kernel (default: 2)
        :param strides          : List of stride that are applied when moving to the next stage
                                  (default: [2, 2, 2, 2, 2, 2]) (NOTE: also include the first conv and maxpool as stage)
        :param dilations        : List of dilations that are applied upon each stage (default: [1, 1, 1, 1])
        :param downType         : Downsampling method (default: maxpool) [maxpool, conv, avgpool]
        :param lastPool         : Apply a global pooling on last feature map to remove the spatial dimension
        """
        super().__init__()

        # params
        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.archType = archType

        self.build_params()
        self.build_hparams(**kwargs)


    # =============================================== INTERNAL CONFIG =================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        return {
            'norm'            : 'BatchNorm2d',      'normKwargs'      : {},
            'activ'           : 'ReLU',             'activKwargs'     : {'inplace' : True},
            'lastActiv'       : None,               'lastActivKwargs' : {},
            'firstLayerKernel': 7,                  'lastLayerKernel' : 1,
            'kernel'          : 3 ,                 'poolKernel'      : 2,
            'strides'         : [2, 2, 2, 2, 2, 2], 'dilations'       : [1, 1, 1, 1],
            'downType'        : 'maxpool',          'lastPool'        : True
        }

    # ================================================== ARCH ==========================================================
    @property
    def architectures(self):
        return {'18'  : [2, 2, 2, 2] , '34' : [3, 4, 6, 2]}

    # =============================================== BUILD ============================================================
    def build(self):
        """
        Call this for initializing the modules of the network.
        :return:
        """
        l = self.architectures[str(self.archType)]

        layers = nn.ModuleList()

        # step 1 - create first conv and pool
        layers.add_module('first', Conv2dBlock(self.inChNo, 64, self.firstLayerKernel, stride=self.strides[0],
                                               activType=self.activ, activKwargs=self.activKwargs,
                                               normType=self.norm, normKwargs=self.normKwargs))

        layers.add_module('first_pool', self._get_down_layer(64, self.strides[1]))

        # step 2 - create the 4 stages
        layers.add_module('stage_1', self._make_stage(l[0], inChNo=64, hidChNo=64, outChNo=64,
                                                      stride=self.strides[2], dilation=self.dilations[0]))
        layers.add_module('stage_2', self._make_stage(l[1], inChNo=64, hidChNo=128, outChNo=128,
                                                      stride=self.strides[3], dilation=self.dilations[1]))
        layers.add_module('stage_3', self._make_stage(l[2], inChNo=128, hidChNo=256, outChNo=256,
                                                      stride=self.strides[4], dilation=self.dilations[2]))
        layers.add_module('stage_4', self._make_stage(l[3], inChNo=256, hidChNo=512, outChNo=512,
                                                      stride=self.strides[5], dilation=self.dilations[3]))

        # step 3 - add last layer
        if self.outChNo is not None or self.outChNo == 0:
            layers.add_module('classif', Conv2dBlock(512, self.outChNo, self.lastLayerKernel,
                                                  activType = self.lastActiv, activKwargs = self.lastActivKwargs,
                                                  normType  = self.norm,      normKwargs  = self.normKwargs))
        # step 4 - remove spatial dimensions
        if self.lastPool:
            layers.add_module('last_pool', nn.AdaptiveAvgPool2d(1))

        self.model = layers
        return self

    # =============================================== MAKE STAGES ======================================================
    def _make_stage(self, noBlocks, inChNo, hidChNo, outChNo, stride, dilation):
        """
        Create a stage of repeated Residual Blocks

        :param noBlocks:  no of residual blocks to be repeated
        :param inChNo:    number if input channels in the stage
        :param hidChNo:   number of hidden channels in the stage
        :param outChNo:   number of output channels to the next stage
        :param stride:    apply this stride on the first ResBlock
        :param dilation:  apply this dilation
        :return: nn.Sequantial
        """
        layers = list()

        # first layer is always different and here is where the stride is applied
        layers += [ResBlock(inChNo, outChNo, hidChNo, stride = stride, kernel=self.kernel,
                            activType = self.activ,    activKwargs = self.activKwargs,
                            normType  = self.norm, normKwargs  = self.normKwargs)]

        # add the rest of the blocks
        for i in range(1, noBlocks):
            layers += [ResBlock(outChNo, outChNo, hidChNo, dilation = dilation, kernel=self.kernel,
                                activType = self.activ,    activKwargs = self.activKwargs,
                                normType  = self.norm, normKwargs  = self.normKwargs)]

        return nn.Sequential(*layers)

    # =============================================== GET DOWN LAYER ===================================================
    def _get_down_layer(self, inChNo, stride=2):
        """
        Return the down layer.
        :param inChNo: number of input channels
        :param stride: the stride
        :return:
        """
        if self.downType == 'conv':
            return Conv2dBlock(inChNo, inChNo, self.poolKernel, stride=stride, activType=None, normType=None)

        if self.downType == 'maxpool':
            return nn.MaxPool2d(self.poolKernel, stride=stride, padding=(self.poolKernel - 1) // 2)

        if self.downType == 'avgpool':
            return nn.AvgPool2d(self.poolKernel, stride=stride, padding=(self.poolKernel - 1) // 2)

        raise ValueError('Downtype [%s] not implemented. Choose from [conv/maxpool/avgpool]' % self.downType)

    # ================================================== INIT NETWORK ==================================================
    def init_weights(self):
        """
        Initialize the models weights with defaults.
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.activ.lower())
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        return self

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        """
        Forward function of ResNet encoder.
        :param inTensor: input tensor [B, C, H, W]
        :return:
        """
        b, c, h, w = inTensor.shape
        for name, module in self.model.named_children():
            inTensor = module(inTensor)

        if self.lastPool:
            inTensor = inTensor.reshape(b, self.outChNo)

        return inTensor


# =============================================== MAIN =================================================================
if __name__ == '__main__':
    import torch

    # ====> Task 2 <===
    # One can test the initialization of ResNet here

    # this (very) dummy initialization, please modify
    temp = torch.randn((4, 69, 128, 128)) # mimic a batch of 4 images [B C H W]
    resnet = ResNet(
            inChNo           = 69,                 outChNo         = 666,
            norm             = 'BatchNorm2d',      normKwargs      = {},
            activ            = 'SiLU',             activKwargs     = {},
            lastActiv        = 'GELU',             lastActivKwargs = {},
            strides          = [1, 1, 1, 1, 1, 1], downType        = 'avgpool',
            lastPool         = False,              archType        = '18',
            firstLayerKernel = 7,                  lastLayerKernel = 1).build()

    outTensor = resnet(temp)
    print(outTensor.shape)
