from torch import nn
from torch.nn import Dropout, Linear
from core.nn.blocks.utils import simple_import

# ######################################################################################################################
#                                                 LINEAR BLOCK
# ######################################################################################################################
class LinearBlock(nn.Module):
    def __init__(self, inChNo,        outChNo,
                 activType = 'ReLU',  activKwargs = {},
                 normType  = None,    normKwargs  = {},
                 dropRate  = 0.0,     **kwargs):
        super().__init__()

        model = nn.ModuleList()

        # get activation and normalization layer
        norm  = simple_import(normType,   kwargs.get('normPack',  'torch.nn'))
        activ = simple_import(activType,  kwargs.get('activPack', 'torch.nn'))

        # only use bias when no normalization is used
        if kwargs.get('bias', None) is None:
            kwargs['bias'] = True if norm is None else False

        model.append(Linear(inChNo, outChNo, bias=kwargs['bias']))
        if norm  is not None:
            model.append(norm(**normKwargs))
        if activ is not None:
            model.append(activ(**activKwargs))
        if dropRate >= 0:
            model.append(Dropout(dropRate))

        self.model = model

    # ================================================== FORWARD ==================================================
    def forward(self, inTensor):
        for name, module in self.model.named_children():
            inTensor = module(inTensor)

        return inTensor

# ================================================== TEST ==================================================
if __name__ == '__main__':
    import torch
    block  = LinearBlock(10, 10, 0, 'ReLU')
    tensor = torch.randn(10)

    block(tensor)