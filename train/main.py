import torch
import argparse
import numpy as np

from tqdm               import tqdm
from train.config       import cfg
from tensorboardX       import SummaryWriter
from train.trainer      import SimCLRTrainer
from core.nn.utils.misc import create_logger

# ================================================== RANDOMNESS ========================================================
argparser = argparse.ArgumentParser(description='Training script fot road lines segmentation models')
argparser.add_argument('-c', type=str,
                       help    = 'Path to the configuration file',
                       default = 'assets/experiments/base_simclr.yaml',)

# remove randomness
np.random.seed(13)
torch.manual_seed(13)
torch.cuda.manual_seed(13)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.reproducible  = True
torch.backends.cudnn.benchmark     = True

# ================================================== MAIN ==============================================================
if __name__ == '__main__':
    args = argparser.parse_args()
    if args.c is not None and args.c != '':
        cfg.merge_from_file(args.c)

    # step 1 - loggers
    writer = SummaryWriter(logdir=cfg.general.logsDir)
    logger = create_logger(name=cfg.general.name, saveDir=cfg.general.logsDir, logLevel=cfg.log.logLevel)
    logger.info(f'Epoch {cfg.train.loadEpoch}: start training ')
    logger.info(str(cfg))

    # step 2 - initialize models
    trainer = SimCLRTrainer(cfg, logger=logger)
    dataLen = len(trainer.trainData)
    valLen  = len(trainer.valData)

    # step 3 - training loop
    for epoch in range(cfg.train.loadEpoch, cfg.train.noEpochs):
        logger.info(f"Epoch {epoch}/{cfg.train.noEpochs}")
        pBar = tqdm(trainer.trainData, total=dataLen)

        # step 3.1 - train for one epoch
        for step, data in enumerate(pBar):

            # step 3.1.1 - optimize parameters
            lossDict = trainer.optimize_parameters(*data)
            if lossDict is None:
                continue

            # step 3.1.2 - log loss and lr
            if step % cfg.log.infoSteps == 0:
                pBar.set_postfix({'|| Loss': ' %2.5f ' % lossDict['Loss']})
                writer.add_scalars('train/loss', lossDict, epoch*dataLen + step)

                lrDict = trainer.get_lr()
                writer.add_scalars('train/lr',   lrDict,   epoch*dataLen + step)

        # step 3.2 - save model
        if (epoch % cfg.log.saveEpoch == 0) or (epoch == (cfg.train.noEpochs - 1)):
            trainer.save_models(epoch)
            trainer.save_optimizers(epoch)

        # step 3.3 - eval model
        if (epoch % cfg.log.evalEpoch == 0) or (epoch == (cfg.train.noEpochs - 1)):
            eBar = tqdm(trainer.valData, total=valLen)
            allEmbbds = []
            allImgs   = []
            allLabels = []

            # step 3.3.1 - get embeddings for all eval set
            for step, data in enumerate(eBar):
                embbed, normImages = trainer.evaluate(*data)
                allEmbbds.append(embbed)
                allImgs.append(normImages)
                allLabels.append(data[1].reshape(-1, 1))

            allEmbbds = torch.cat(allEmbbds, dim=0)
            allImgs   = torch.cat(allImgs,   dim=0)
            allLabels = torch.cat(allLabels, dim=0)

            # step 3.3.2 - send embeddings to TensorBoard
            writer.add_embedding(allEmbbds, allLabels.tolist(), allImgs, global_step=epoch*dataLen)

        # torch.cuda.empty_cache()
