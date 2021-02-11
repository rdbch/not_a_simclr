import os
import logging
from yacs.config import CfgNode as CN


cfg = CN()

# ================================================== GENERAL ===========================================================
cfg.general             = CN()
cfg.general.name        = 'SimClrTest'                   # Name your experiment
cfg.general.saveDir     = os.path.join('assets', 'checkpoints', cfg.general.name) # Where to save the checkppoints
cfg.general.logsDir     = os.path.join('assets', 'logs'       , cfg.general.name) # Where to save logs
cfg.general.device      = 'cuda'                         # As if one would train this on cpu... ... ... ...
cfg.general.cudaDevices = ['0']                          # If one has multiple GPUs, select from them

# ================================================== TRAIN =============================================================
cfg.train              = CN()
cfg.train.loadEpoch     = 0                              # Load epoch (used when resuming)
cfg.train.noEpochs     = 100                             # Number of training epochs

# ================================================== DATA ==============================================================
cfg.data = CN()
cfg.data.dataset       = 'CIFAR10'                       # Name from torchvision, available [CIFAR10]
cfg.data.rootPath      = 'D:/Datasets/CULane'            # Dir to save the dataset

cfg.data.trainBatch    = 256                             # Training batch size    [256 should fit in 8GB RTX2080]
cfg.data.valBatch      = 256                              # Validation batch size  [256 should fit in 8GB RTX2080]

cfg.data.trainWorkers  = 6                               # No of threads to fetch train data [dependent on your CPU]
cfg.data.valWorkers    = 6                               # No of threads to fetch   val data [dependent on your CPU]

cfg.data.trainSize     = (128, 128)                      # Training size (too small can decrease ResNet performance)
cfg.data.valSize       = (128, 128)                      # Validation size (too small can decrease ResNet performance)

# ================================================== MODEL =============================================================
cfg.model = CN()
cfg.model.kwargs = CN(new_allowed=True)
cfg.model.kwargs.inChNo          = 3                     # RGB => 3 channels
cfg.model.kwargs.latentChNo      = 512                   # Latent size H
cfg.model.kwargs.outChNo         = 64                    # Latent size of Z
cfg.model.kwargs.encoderStrides  = [2, 2, 1, 1, 2, 2]    # At each 2, the spatial dimension will be halfed
cfg.model.kwargs.mlpLayers       = [128, 128]            # Layers in the MLP (projection function)
cfg.model.kwargs.norm            = 'BatchNorm2d'         # Normalization layer for encoder
cfg.model.kwargs.normKwargs      = CN(new_allowed=True)  # Kwargs passed to it
cfg.model.kwargs.activ           = 'ReLU'                # Activation function used in encoder
cfg.model.kwargs.activKwargs     = CN(new_allowed=True)  # Kwargs passed to it
cfg.model.kwargs.lastActiv       = None                  # Activation function used in encoder
cfg.model.kwargs.lastActivKwargs = CN(new_allowed=True)  # Kwargs passed to it
cfg.model.kwargs.dropRate        = None                  # Dropout rate for the MLP layers
cfg.model.kwargs.dropLayers      = 1                     # Apply dropout on this last layers

# ================================================== LOSSES ============================================================
cfg.loss = CN()
cfg.loss.temperature = 0.5                               # Temperature of the loss function

# ================================================== OPTIMIZER =========================================================
cfg.optimizer = CN()
cfg.optimizer.optimName   = 'Adam'                       # Optimizer name from PyTorch [defaulf, Adam]
cfg.optimizer.lr          = 0.0003                       # Small learning rate 3e-4
cfg.optimizer.optimKwargs = CN(new_allowed=True)         # Kwargs passed to it


cfg.optimizer.schedPack   = 'torch.optim.lr_scheduler'   # Learning rate scheduler package
cfg.optimizer.schedName   = 'StepLR'                     # Learning rate scheduler name
cfg.optimizer.schedKwargs = CN(new_allowed=True)         # Kwargs passed to it

# ================================================== LOGGING ===========================================================
cfg.log = CN()

cfg.log.logLevel    = logging.INFO                       # Logging module stuff
cfg.log.infoSteps   = 20                                 # Update Tqdm bar (measure in no of batches)
cfg.log.saveEpoch   = 1000                               # save the model every saveSteps
cfg.log.evalEpoch   = 1000                               # Log embeddings to Tensorboard

# res = cfg.dump()
# with open(cfg.general.name + '.yaml', 'w') as f:
#     f.write(res)