from torch import nn

from core.nn.blocks.conv2d_block import Conv2dBlock
from core.nn.blocks.resblock import ResBlock
from core.nn.utils.configurable import Configurable


########################################################################################################################
#                                               RESNET
########################################################################################################################
class ResNet(nn.Module, Configurable):
    def __init__(self, inChNo=3, outChNo = None, archType = '18', **kwargs):
        super().__init__()

        # params configuration
        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.archType = archType

        self.build_params()
        self.build_hparams(**kwargs)

    # ================================================== ARCH ==========================================================
    @property
    def architectures(self):
        return {'18'  : [2, 2, 2, 2] , '34' : [3, 4, 6, 2]}

    # =============================================== INTERNAL CONFIG =================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        return {
            'norm'            : 'BatchNorm2d',      'normKwargs'      : {},
            'activ'           : 'ReLU',             'activKwargs'     : {'inplace' : True},
            'lastActiv'       : None,               'lastActivKwargs' : {},
            'firstLayerKernel': 7,                  'lastLayerKernel' : 1,
            'kernel'          : 3 ,                 'poolKernel'      : 2,
            'strides'         : [2, 2, 1, 1, 2, 2], 'dilations'       : [1, 1, 1, 1],
            'downType'        : 'maxpool',          'lastPool'        : True
        }

    # =============================================== BUILD ============================================================
    def build(self):
        l = self.architectures[str(self.archType)]
        layers = nn.ModuleList()
        layers.add_module('first', Conv2dBlock(self.inChNo, 64, self.firstLayerKernel, stride=self.strides[0],
                                               activType=self.activ, activKwargs=self.activKwargs,
                                               normType=self.norm, normKwargs=self.normKwargs))

        layers.add_module('first_pool', self._get_down_layer(64, self.strides[1]))

        # the stage values are taken from the original paper,
        layers.add_module('stage_1', self._make_stage(l[0], inChNo=64, hidChNo=64, outChNo=64,
                                                      stride=self.strides[2], dilation=self.dilations[0]))
        layers.add_module('stage_2', self._make_stage(l[1], inChNo=64, hidChNo=128, outChNo=128,
                                                      stride=self.strides[3], dilation=self.dilations[1]))
        layers.add_module('stage_3', self._make_stage(l[2], inChNo=128, hidChNo=256, outChNo=256,
                                                      stride=self.strides[4], dilation=self.dilations[2]))
        layers.add_module('stage_4', self._make_stage(l[3], inChNo=256, hidChNo=512, outChNo=512,
                                                      stride=self.strides[5], dilation=self.dilations[3]))


        if self.outChNo is not None or self.outChNo == 0:
            layers.add_module('classif', Conv2dBlock(512, self.outChNo, self.lastLayerKernel,
                                                  activType = self.lastActiv, activKwargs = self.lastActivKwargs,
                                                  normType  = self.norm,      normKwargs  = self.normKwargs))

        if self.lastPool:
            layers.add_module('last_pool', nn.AdaptiveAvgPool2d(1))

        self.model = layers

        return self

    # =============================================== MAKE STAGES ======================================================
    def _make_stage(self, noBlocks, inChNo, hidChNo, outChNo, stride, dilation):
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

        if self.downType == 'conv':
            return Conv2dBlock(inChNo, inChNo, self.poolKernel, stride=stride, activType=None, normType=None)

        if self.downType == 'maxpool':
            return nn.MaxPool2d(self.poolKernel, stride=stride, padding=(self.poolKernel - 1) // 2)

        if self.downType == 'avgpool':
            return nn.AvgPool2d(self.poolKernel, stride=stride, padding=(self.poolKernel - 1) // 2)

        raise ValueError('Downtype [%s] not implemented. Choose from [conv/maxpool/avgpool]' % self.downType)

    # ================================================== INIT NETWORK ==================================================
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.activ.lower())
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        b, c, h, w = inTensor.shape
        for name, module in self.model.named_children():
            inTensor = module(inTensor)

        if self.lastPool:
            inTensor = inTensor.reshape(b, self.outChNo)

        return inTensor


# =============================================== MAIN =================================================================
if __name__ == '__main__':
    import torch

    resnet = ResNet(inChNo=3, outChNo = None, archType = '18').build()
    resnet.default_init()
    temp   = torch.randn((4, 3, 512, 512))

    outTensor = resnet(temp)

    print(outTensor.shape)
