import importlib

# ================================================== SAME PADDING ======================================================
def same_padding(kSize, dilSize):
    """
    Function that return the the paremeters for obtaining the "same" padding type.

    :param kSize:   int denoting the kernel size
    :param dilSize: int denoting the dilation rate of the conv block
    :return:
    """

    kSizeEffective = kSize + (kSize - 1) * (dilSize - 1)
    padTotal       = kSizeEffective - 1
    padStart       = padTotal // 2
    padEnd         = padTotal - padStart

    return padStart, padEnd, padStart, padEnd

# ================================================== SIMPLE IMPORT =====================================================
def simple_import(name, pack):
    """
    Import a module by name and return a given sub-module.

    :param pack: the name of the module
    :param name: name of the submodule to be returned
    :return:
    """
    if name is not None:
        return getattr(importlib.import_module(pack, name), name)

    return None