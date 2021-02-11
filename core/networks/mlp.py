import torch
from torch import nn

from core.nn.blocks.linear_block import LinearBlock
from core.nn.utils.configurable import Configurable


# ================================================== MLP ===============================================================
class MLP(nn.Module, Configurable):
    def __init__(self, inChNo=64, outChNo=None, layerCfg=(64, 64), **kwargs):
        '''
        Multi layer perceptron class
        :param inChNo           : number of input features
        :param outChNo          : number of output features (if None, use layerCfg[-1])
        :param layerCfg         : a list containing the number of neurons for each layer
        :param kwargs
        :param activ            : activation used on the output of each layer
        :param activKwargs      : kwargs for activ
        :param activLast        : activation used on the last layer
        :param activLastKwargs  : kwargs for activLast
        :param normType         : normalization type (name from torch)
        :param normKwargs       : kwargs passed to normalization
        :param dropRate         : dropout rate
        :param dropLayers       : only apply dropout on the last dropLayers
        '''
        super().__init__()

        self.inChNo   = inChNo
        self.outChNo  = outChNo
        self.layerCfg = layerCfg

        self.build_hparams(**kwargs)

    # =============================================== INTERNAL CONFIG ==================================================
    @Configurable.hyper_params()
    def build_hparams(self, **kwargs):
        ''' This method sets up the hyperparameters. Default values are provided.'''

        return {
            'norm'      :  None,  'normKwargs'      : {},
            'activ'     : 'ReLU', 'activKwargs'     : {},
            'lastActiv' :  None,  'lastActivKwargs' : {},
            'dropRate'  :  None,  'dropLayers'      : len(self.layerCfg),
            'useBias'   :  False,
        }

    # =============================================== BUILD ============================================================
    def build(self):
        '''
        Build the layer configuration of the components.
        :return: self
        '''

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
        if self.dropRate is not None:
            if self.dropRate > 0:
                if i > (len(self.layerCfg) - self.dropLayers):
                    return True

        return False

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        '''
        :param inTensor: input tensor
        :return: prediction
        '''
        for module in self.model:
            inTensor = module(inTensor)

        return inTensor

# ================================================== MAIN ==============================================================
if __name__ == '__main__':
    mlp = MLP(100, 10, (63, 64, 65)).build()
    tensor = torch.randn(10,100)

    res = mlp(tensor)
    print(res.shape)