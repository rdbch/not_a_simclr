from torch import nn
from torch.nn.utils import spectral_norm
from core.nn.blocks.utils import simple_import, same_padding

# ######################################################################################################################
#                                               CONV2D BLOCK
# ######################################################################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, inChNo, outChNo, kernel, padType = 'ReflectionPad2d', spectral = False,
                 activType = 'ReLU',  activKwargs = {},
                 normType  = None,    normKwargs  = {},
                 **kwargs):
        '''
        Convolution block. A simple class that groups the common structure  pad -> conv -> norm -> relu sequence.
        It is to be noted that by default, if a normalization layer is used, the bias is disabled by default(it can be
        overwritten)
        :param inChNo:      number of input channels
        :param outChNo:     number of output channels
        :param kernel:      the kernel size for the convolution layer
        :param padType:     padding type (default: ReflectionPad2d)
        :param padPack:     the package where the padding can be found (default: torch.nn)
        :param activType:   activation type (default: ReLU)
        :param activKwargs: dictionary of arguments to be passed to the activation layer
        :param activPack:   the package where the activation layer can be found (default: torch.nn)
        :param normType:    normalization type (default: BatchNorm2d)
        :param normKwargs:  dictionary of arguments to be passed to the normalization layer
        :param normPack:    the package where the normalization layer can be found (default: torch.nn)
        :param spectral:    apply spectral normalization to the convolution layer
        :param kwargs:      any are keyword arguments will be passed to the convolution layer
        '''
        super().__init__()

        # import normalization layer and activation
        norm    = simple_import(normType,   kwargs.get('normPack',  'torch.nn'))
        activ   = simple_import(activType,  kwargs.get('activPack', 'torch.nn'))
        padding = simple_import(padType,    kwargs.get('padPack',   'torch.nn'))

        # disable bias by default if normalization is used
        if kwargs.get('bias', None) is None:
            kwargs['bias'] = True if norm is None else False

        # compute the padding value
        padValue  = same_padding(kernel, kwargs.get('dilation', 1))

        # make the structure pad->conv->norm->activ
        layers = list()

        # step 1 - pad layer
        if padding is not None:
            layers.append(padding(padValue))

        # step 2 - convolutional layer
        if spectral:
            layers.append(spectral_norm(nn.Conv2d(inChNo, outChNo, kernel_size=kernel, **kwargs)))
        else:
            layers.append(nn.Conv2d(inChNo, outChNo, kernel_size=kernel, **kwargs))

        # step 3 - normalization (batch/instance/adain/etc)
        if norm is not None:
            layers.append(norm(outChNo, **normKwargs))

        # step 4 - activation / non-linearity function (ReLU/Tanh/etc)
        if activ is not None:
            layers.append(activ(**activKwargs))

        self.model = nn.Sequential(*layers)

    # =============================================== FORWARD ==========================================================
    def forward(self, inTensor):
        return self.model(inTensor)