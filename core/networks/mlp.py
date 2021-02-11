import torch
from torch import nn

from core.nn.blocks             import LinearBlock
from core.nn.utils.configurable import Configurable

# ######################################################################################################################
#                                                        MLP
# ######################################################################################################################
class MLP(nn.Module, Configurable):
    def __init__(self, inChNo=64, outChNo=None, layerCfg=(64, 64), **kwargs):
        """
        Multi layer perceptron class.

        It is a module list  of LinearBlocks
        LinearBlock = LinearUnit + Normalization + Activation

        Example:
            mlp = MLP(inChNo=100, outChNo=10, layerCfg=(64, 64, 64)).build()
            res = mlp(tensor)

        :param inChNo          : Number of of input features (channels)
        :param outChNo         : Number of output features (if None, use layerCfg[-1])
        :param layerCfg        : List containing the number of neurons for each layer
        :param activ           : Activation used on the output of each layer *
        :param activKwargs     : Kwargs for activ (see Pytorch docs)
        :param activLast       : Activation applied on the last layer *
        :param activLastKwargs : Kwargs for the last activ (see Pytorch docs)
        :param norm            : Normalization layer name *
        :param normKwargs      : Kwargs passed to normalization layer
        :param dropRate        : Dropout rate [0, 1]
        :param dropLayers      : Apply dropout only on the last dropLayers

        * Use a string with the PyTorch name (i.e. BatchNorm2d, ReLU, etc)
        """
        super().__init__()

        # params
        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.layerCfg = layerCfg

        self.build_params()
        self.build_hparams(**kwargs)

    # =============================================== INTERNAL CONFIG ==================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        # hparams
        return {
            'norm'      :  None,  'normKwargs'      : {},
            'activ'     : 'ReLU', 'activKwargs'     : {},
            'lastActiv' :  None,  'lastActivKwargs' : {},
            'dropRate'  :  None,  'dropLayers'      : len(self.layerCfg),
            'useBias'   :  False,
        }

    # =============================================== BUILD ============================================================
    def build(self):
        """
        Call this for initializing the modules of the network.
        :return:
        """

        model = nn.ModuleList()

        # create layer structure
        for i, layerCfg in enumerate(self.layerCfg):
            prevChNo = self.inChNo if i == 0 else self.layerCfg[i-1]

            dropRate = self.dropRate if self.__use_droput(i) else 0.0
            model.append(LinearBlock(prevChNo, layerCfg,
                                     normType  = self.norm,     normKwargs  = self.normKwargs,
                                     activType = self.activ,    activKwargs = self.activKwargs,
                                     useBias   = self.useBias , dropRate    = dropRate))

        # add last layer
        if self.outChNo is not None:
            prevChNo = self.layerCfg[-1] if len(self.layerCfg) > 0 else self.inChNo
            model.append(LinearBlock(prevChNo, self.outChNo,
                                     normType  = None,           normKwargs  = {},
                                     activType = self.lastActiv, activKwargs = self.lastActivKwargs,
                                     useBias   = self.useBias ,  dropRate    = 0.0))

        self.model = model
        return self

    # =============================================== CHECK DROPOUT ====================================================
    def __use_droput(self, i):
        """
        Compute if we want to apply DropOut to a layer or not.
        :param i: the current layer index
        :return: bool
        """
        if self.dropRate is not None:
            if self.dropRate > 0:
                if i > (len(self.layerCfg) - self.dropLayers):
                    return True

        return False

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        """
        :param inTensor: input tensor [B, N]
        :return:
        """

        for name, module in self.model.named_children():
            inTensor = module(inTensor)

        return inTensor

# ================================================== MAIN ==============================================================
if __name__ == '__main__':
    mlp = MLP(100, 10, (63, 64, 65)).build()
    tensor = torch.randn(10,100)

    res = mlp(tensor)
    print(res.shape)