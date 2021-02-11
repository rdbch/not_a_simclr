from torch import nn

from core.networks.mlp          import MLP
from core.networks.resnet       import ResNet
from core.nn.utils.configurable import Configurable

# ######################################################################################################################
#                                                   SIMCLR
# ######################################################################################################################
class SimCLR(nn.Module, Configurable):
    def __init__(self, inChNo, latentChNo, outChNo, **kwargs):
        super().__init__()
        self.inChNo     = inChNo
        self.latentChNo = latentChNo
        self.outChNo    = outChNo

        self.build_params()
        self.build_hparams(**kwargs)

    # ================================================== BUILD HPARAMS =================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        return {
            'norm'           : 'BatchNorm2d',      'normKwargs'      : {},
            'activ'          : 'ReLU',             'activKwargs'     : {'inplace' : True},
            'encoderStrides' : [2, 2, 1, 1, 2, 2], 'mlpLayers'       : [128, 128],
            'dropRate'       :  None,              'dropLayers'      : 1,
            'archType'       : '18'
        }

    # ================================================== BUILD =========================================================
    def build(self):
        self.encoder = ResNet(inChNo   = self.inChNo,         outChNo     = self.latentChNo,
                              norm     = self.norm,           normKwargs  = self.normKwargs,
                              activ    = self.activ,          activKwargs = self.activKwargs,
                              strides  = self.encoderStrides, downType    = 'maxpool',
                              lastPool = True,                archType    = self.archType,
                              ).build()

        self.project = MLP(inChNo   = self.latentChNo, outChNo     = self.outChNo,
                           layerCfg = self.mlpLayers,  useBias     = False,
                           norm     = None,            normKwargs  = {},
                           activ    = self.activ,      activKwargs = self.activKwargs,
                           dropRate = self.dropRate,   dropLayers  = self.dropLayers
                           ).build()
        return self

    # ================================================== FORWARD =======================================================
    def forward(self, inTensor):
        h = self.encoder(inTensor)
        z = self.project(h)

        return h, z

    # ================================================== INIT WEIGHTS ==================================================
    def init_weights(self):
        self.encoder.init_weights()

    # ================================================== GET PARAMS ====================================================
    def get_params(self, baseLr):
        return [{'params' : self.parameters(), 'lr' : baseLr}]