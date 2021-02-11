import torch
import importlib

import torchvision.datasets
from torch.utils.data            import DataLoader
from core.nn.loss                import NTXent
from core.networks.simclr        import SimCLR
from core.nn.utils.base_trainer  import BaseTrainer
from core.data.data_augmentation import ImageTransforms, IMAGENER_STD, IMAGENET_MEAN

# ######################################################################################################################
#                                               LANE SEGMENTATION TRAINER
# ######################################################################################################################
class SimCLRTrainer(BaseTrainer):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

    # ================================================== INIT DATA =====================================================
    def init_train_data(self, cfg):
        c = cfg.data

        dataset  = getattr(torchvision.datasets, c.dataset)
        imgTrans = ImageTransforms(c.trainSize)

        # train data
        trainDataset   = dataset(root=c.rootPath, train=True, transform=imgTrans, download=True)
        self.trainData = DataLoader(trainDataset, batch_size=c.trainBatch, num_workers=c.trainWorkers, pin_memory=True)
        self.trainDataset = trainDataset

        # eval data
        valDataset    = dataset(root=c.rootPath, train=False, transform=imgTrans, download=True)
        self.valData  = DataLoader(valDataset, batch_size=c.valBatch, num_workers=c.valWorkers, pin_memory=True)
        self.trainDataset = valDataset

    # =============================================== REGISTER MODEL ===================================================
    def init_networks(self, cfg):
        c = cfg.model

        self.network = SimCLR(**dict(c.kwargs)).build()
        self.nameModels.append('network')

        # set-up weights
        self.network.init_weights()
        if cfg.train.loadEpoch > 0:
            self.load_models(cfg.train.loadEpoch)

        # copy the model to device
        self.network.to(cfg.general.device)

    # =============================================== REGISTER LOSS ====================================================
    def init_losses(self, cfg):
        c = cfg.loss

        self.loss = NTXent(cfg.data.trainBatch, c.temperature)

    # =============================================== REGISTER OPTIMIZER ===============================================
    def init_optimizers(self, cfg):

        c = cfg.optimizer

        # optimizer
        params         = self.network.get_params(c.lr)
        self.optimizer = getattr(torch.optim, c.optimName)(params, **c.optimKwargs)
        self.nameOptims.append('optimizer')

        if cfg.train.loadEpoch != 0:
            self.load_optimizers(cfg.train.loadIter)

        # scheduler
        if c.schedName is not None:
            load      = cfg.train.loadEpoch if cfg.train.loadEpoch != 0 else -1
            schedPack = importlib.import_module(c.schedPack)
            sched     = getattr(schedPack, c.schedName)(self.optimizer, last_epoch=load, **cfg.optimizer.schedKwargs)

            sched.last_epoch = cfg.train.loadEpoch
            self.scheduler   = sched

    # ================================================== OPTIMIZE PARAMS ===============================================
    def optimize_parameters(self, *args, **kwargs):

        c = self.cfg

        # step 1 - copy the data to the specific device
        augImages   = args[0][0].to(c.general.device)
        plainImages = args[0][1].to(c.general.device)

        if augImages.shape[0] < self.cfg.data.trainBatch:
            return None

        # step 2 - reset gradients
        self.optimizer.zero_grad()
        self.network.zero_grad()

        # step 3 - get representations
        hA, zA  = self.network(augImages)
        hB, zB  = self.network(plainImages)

        # step 4 - compute losses
        loss = self.loss(zA, zB)

        # step 5 - backprop
        loss.backward()
        self.optimizer.step()

        # step 6 - update learning rate
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        return {'Loss' : loss.item()}

    # ================================================== EVALUATE RESULTS ==============================================
    def evaluate(self, *args, **kwargs):
        self.network.eval()
        plainImages = args[0][1].to(self.cfg.general.device)

        with torch.no_grad():
            h, _        = self.network(plainImages)
            h           = h.cpu()
            plainImages = plainImages.cpu().mul_(IMAGENER_STD).add_(IMAGENET_MEAN).clamp(0, 1)
            plainImages = torch.nn.functional.interpolate(plainImages, (32, 32)) # max size allowed by Tensorboard

        self.network.train()
        return h, plainImages

    # ================================================== GET LR ========================================================
    def get_lr(self):
        """
        Return the learning rates registered in the optimizer.
        :return:  a dict with the learning rates
        """

        lrs = {}
        for i, pg in enumerate(self.optimizer.param_groups):
            lrs[f'lr_{i}'] = pg['lr']

        return lrs



# ================================================== TEST ==============================================================
if __name__ == '__main__':
    input = torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    target =torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    print(input)
    print(target)

