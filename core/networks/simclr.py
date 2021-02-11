from torch import nn

from core.networks              import MLP, ResNet
from core.nn.utils.configurable import Configurable

# ######################################################################################################################
#                                                   SIMCLR
# ######################################################################################################################
class SimCLR(nn.Module, Configurable):
    def __init__(self, inChNo, latentChNo, outChNo, **kwargs):
        """
        Wrapper class for the SimCLR method. This network is composed from 2 main networks:
            - An encoder :  ResNet encoder used for extracting a latent representations of data
            - A projector:  A MLP that projects the the latent representation from the encoder, to another latent
                            space where a contrastive loss can be applied

        Example:
            model = SimCLR(inChNo=2, latentChNo=512, outChNo=256, norm='BatchNorm2d').build()
            h, z  = model(tensor)

        :param inChNo         : Number of input channels
        :param latentChNo     : Latent space dimension (H)
        :param outChNo        : Latent space dimension (Z)
        :param archType       : The architecture if the encoder (default: '18') ['18', '34']
        :param activ          : Activation used into encoder and projection function (default: ReLU)*
        :param activKwargs    : Kwargs for activ (default: {'inplace' : True})
        :param norm           : Normalization layer name used in encoder * (default: BatchNorm2d)
        :param normKwargs     : Kwargs passed to normalization layer in the encoder(default: {})
        :param encoderStrides : List of stride that are applied when moving to the next stage
                                (default: [2, 2, 2, 2, 2, 2]) (NOTE: also include the first conv and maxpool as stage)
        :param mlpLayers      : List containing the number of neurons for each layer in the projector
                                (default: (128, 128)
        :param dropRate       : Dropout rate in the MLP
        :param dropLayers     : Apply dropout on the last layers in MLP
        """

        super().__init__()

        # params
        self.inChNo     = inChNo
        self.latentChNo = latentChNo
        self.outChNo    = outChNo

        self.build_params()
        self.build_hparams(**kwargs)

    # ================================================== BUILD HPARAMS =================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        return {
            'norm'           : 'BatchNorm2d',      'normKwargs'  : {},                  # <=== USE THIS FOR 2 AND 3
            'activ'          : 'ReLU',             'activKwargs' : {'inplace' : True},
            'encoderStrides' : [2, 2, 1, 1, 2, 2], 'mlpLayers'   : [128, 128],
            'dropRate'       :  None,              'dropLayers'  : 1,
            'archType'       : '18'
        }

    # ================================================== BUILD =========================================================
    def build(self):
        """
        Call this for initializing the modules of the network.
        :return:
        """

        # IMPORTANT: Use above hparameters with <<self. >> such that they will be
        # visible from "outside" and could be easily configurable

        # Task 2
        self.encoder = ResNet().build()                                                 # <======= YOUR TASK #2

        # Task 3
        self.project = MLP().build()                                                    # <======= YOUR TASK #3

        return self

    # ================================================== FORWARD =======================================================
    def forward(self, inTensor):
        """
        :param inTensor: input tensor
        :return: h - encoder latent space z - contrastive latent space
        """
        h = self.encoder(inTensor)
        z = self.project(h)

        return h, z

    # ================================================== INIT WEIGHTS ==================================================
    def init_weights(self):
        self.encoder.init_weights()

    # ================================================== GET PARAMS ====================================================
    def get_params(self, baseLr):
        """
        Create the a dictionary of the parameters that are going to be optimized.
        :param baseLr: learning rate to use for parameters
        :return:
        """
        return [{'params' : self.parameters(), 'lr' : baseLr}]


# ================================================== MAIN ==============================================================
if __name__ == '__main__':
    import torch
    model  = SimCLR(inChNo=2, latentChNo=100, outChNo=200, norm='BatchNorm2d').build()
    tensor = torch.randn((4, 3, 128, 128))

    h, z = model(tensor)
    print(h.shape, z.shape)
