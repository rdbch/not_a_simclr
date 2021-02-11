import os
import logging

from yacs.config import CfgNode as CN

cfg = CN()

# ================================================== GENERAL ===========================================================
cfg.general             = CN()
cfg.general.name        = 'SimClrTest'                   # experiment name
cfg.general.saveDir     = os.path.join('assets', 'checkpoints', cfg.general.name) # save checkpoints in this directory
cfg.general.logsDir     = os.path.join('assets', 'logs'       , cfg.general.name) # save checkpoints in this directory
cfg.general.device      = 'cuda'                         # cpu or cuda (ofcourse cuda)
cfg.general.cudaDevices = ['0']                          # devices available for use during training

# ================================================== TRAIN =============================================================
cfg.train              = CN()
cfg.train.loadEpoch     = 0                           # load training state from this point
cfg.train.noEpochs     = 100
# ================================================== DATA ==============================================================
cfg.data = CN()
cfg.data.dataset       = 'CIFAR10'
cfg.data.rootPath      = 'D:/Datasets/CULane'            # path to the dataset root path

# train transforms
cfg.data.trainBatch    = 12                              # training batch (please make sure is divisible with miniBatch)
cfg.data.valBatch      = 12                              # training batch (please make sure is divisible with miniBatch)

cfg.data.trainWorkers  = 6                               # number of threads to be used for loading the training data
cfg.data.valWorkers    = 6                               # number of threads to be used for loading the training data

cfg.data.trainSize     = (260, 768)                      # resize the training image to this size
cfg.data.valSize       = (256, 768)                      # resize the validation images to this size

# ================================================== MODEL =============================================================
cfg.model = CN()
cfg.model.kwargs = CN(new_allowed=True)        # model to be passed to the model
cfg.model.kwargs.inChNo          = 3
cfg.model.kwargs.latentChNo      = 3
cfg.model.kwargs.outChNo         = 3
cfg.model.kwargs.encoderStrides  = [2, 2, 1, 1, 2, 2]
cfg.model.kwargs.mlpLayers       = [128, 128]
cfg.model.kwargs.norm            = 'BatchNorm2d'
cfg.model.kwargs.normKwargs      = CN(new_allowed=True)
cfg.model.kwargs.activ           = 'ReLU'
cfg.model.kwargs.activKwargs     = CN(new_allowed=True)
cfg.model.kwargs.lastActiv       = 'ReLU'
cfg.model.kwargs.lastActivKwargs = CN(new_allowed=True)
cfg.model.kwargs.dropRate        = None
cfg.model.kwargs.dropLayers      = 1

# ================================================== OPTIMIZER =========================================================
cfg.optimizer = CN()
cfg.optimizer.optimName   = 'Adam'                       # optimizer name from pytorch
cfg.optimizer.lr          = 2e-4                         # base learning rate
cfg.optimizer.optimKwargs = CN(new_allowed=True)         # any additional arguments to be passed to the optimizer

# scheduler
cfg.optimizer.schedPack   = 'torch.optim.lr_scheduler'   # learning rate scheduler
cfg.optimizer.schedName   = 'StepLR'                     # taken from DeepLab paper

cfg.optimizer.schedKwargs = CN(new_allowed=True)         # any additional arguments to be passed to the scheduler

# ================================================== LOSSES ============================================================
cfg.loss = CN()

# source loss
cfg.loss.temperature = 0.5
# ================================================== EVAL ==============================================================
cfg.eval = CN()
cfg.eval.iou          = True                             # segmentation evaluation, compute mIoU
cfg.eval.entropy      = True                             # compute the entropy of the predictions
cfg.eval.enableTuEval = False                            # enable TuSimple evalutation mode


cfg.eval.tuEval               = CN(new_allowed=True)
cfg.eval.tuEval.noLanes       = 4                        # max number of lanes
cfg.eval.tuEval.pixelThresh   = 20                       # default values
cfg.eval.tuEval.angleThresh   = 0.85
cfg.eval.tuEval.pointNoThresh = 100                      # min number of points in seg map to be considered a valid lane

# ================================================== VISUALS ===========================================================
cfg.log = CN()

cfg.log.logLevel    = logging.INFO
cfg.log.useComet    = True                               # use commet for logging
cfg.log.infoSteps   = 20                                 # log the loss value every infoSteps
cfg.log.saveEpoch   = 1000                               # save the model every saveSteps
cfg.log.evalEpoch   = 1000                               # evaluate model every evalSteps
cfg.log.visSteps    = 100                                # log new images every visSteps
cfg.log.visualSize  = (198, 410)                         # resize the visualization to this size

# ================================================== COMET =============================================================
cfg.comet             = CN()
cfg.comet.useProxy    = True                             # configure commet for interal use inside Porsche VPN
cfg.comet.user        = "rdbch"                          # username taken from commet
cfg.comet.projectName = "highway-lane"                   # project  taken from commet
cfg.comet.apiKey      = "tHwbs9Yl6b5YPHV6VCScmu6fE"      # api key  taken from commet
cfg.comet.loadExpId   = ''                             # to be completed when resuming a training session

# res = cfg.dump()
# with open(cfg.general.name + '.yaml', 'w') as f:
#     f.write(res)