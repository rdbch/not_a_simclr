import torch
from torch import nn
from core.nn.blocks.utils import simple_import, same_padding

# ######################################################################################################################
#                                                       RESBLOCK
# ######################################################################################################################
class ResBlock(nn.Module):
    def __init__(self, inChNo, outChNo, hidChNo=None, kernel=3, stride=1, lastActiv=True, padType='ReflectionPad2d',
                 activType='ReLU', activKwargs={}, normType=None, normKwargs={}, **kwargs):
        '''
        Bare bones residual block
        1.Pad - conv - norm - act
        2.Pad - conv - norm - act
        :param inChNo:      number of in channels
        :param outChNo:     number of out channels
        :param hidChNo:     the number of inner/hidden channels
        :param kernel:      kernel size
        :param stride:      stride of the first layer
        :param padType:     the pad type, by default "ReflectionPad2d"
        :param activType:   ativation type for conv layers
        :param activKwargs: kwargs to be passed to the parameters
        :param normType:    normalization type
        :param normKwargs:  kwargs dict to be passed to the normalizaiton layer
        :param normPack:    from where to import the normalization
        :param spectral:    from where to import the normalization
        :param kwargs:      other kwargs to be passed to the conv layer
        '''
        super().__init__()

        self.inChNo  = inChNo
        self.outChNo = outChNo
        self.hidChNo = hidChNo if hidChNo is not None else min(inChNo, outChNo)
        self.stride  = stride

        # import normalization layer and activation
        norm    = simple_import(normType,  kwargs.get('normPack',  'torch.nn'))
        activ   = simple_import(activType, kwargs.get('activPack', 'torch.nn'))
        padding = simple_import(padType,   kwargs.get('padPack',   'torch.nn'))

        # add conv blocks
        layers    = list()

        # get padding of 'same' type
        padValue  = same_padding(kernel, kwargs.get('dilation', 1))

        # disable bias by default if normalization is used
        if kwargs.get('bias', None) is None:
            kwargs['bias'] = True if norm is None else False

        # first block of pad - conv - norm - act
        if padding is not None:
            layers.append(padding(padValue))

        # note that the stride only applies here
        layers.append(nn.Conv2d(inChNo, hidChNo, kernel_size=kernel, stride=stride,**kwargs))
        if norm is not None:
            layers.append(norm(hidChNo, **normKwargs))
        if activ is not None:
            layers.append(activ(**activKwargs))

        # second block of pad - conv - norm
        if padding is not None:
            layers.append(padding(padValue))

        layers.append(nn.Conv2d(hidChNo, outChNo, kernel_size=kernel, stride=1, **kwargs))
        if norm is not None:
            layers.append(norm(outChNo, **normKwargs))

        self.model = nn.Sequential(*layers)

        if activ is not None and lastActiv:
            self.lastActiv = activ(**activKwargs)

        # adapt the depth for residual values (when stride is different that 1, or when the in ch differs from out ch)
        if inChNo != outChNo or stride != 1:
            adaptRes = [nn.Conv2d(inChNo, outChNo, kernel_size=1, bias=False, stride=stride)]
            if norm is not None:
                adaptRes.append(norm(outChNo, **normKwargs))
            self.adaptRes = nn.Sequential(*adaptRes)

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        resValue = inTensor
        inTensor = self.model(inTensor)

        if hasattr(self, 'adaptRes'):
            resValue = self.adaptRes(resValue)

        inTensor += resValue

        if hasattr(self, 'lastActiv'):
            inTensor = self.lastActiv(inTensor)

        return inTensor


if __name__ == '__main__':
    a = ResBlock(64, 128, hidChNo=None, kernel=3, padType='ReflectionPad2d', activType='LeakyReLU',
                 activKwargs={'negative_slope':0.02}, normType=None, normKwargs={}, stride = 2)

    print(a)
    myTensor = torch.tensor(()).new_ones(size=(1, 64, 256, 256))
    print(a(myTensor).shape)