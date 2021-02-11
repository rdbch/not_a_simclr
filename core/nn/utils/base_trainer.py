import os
import torch

# ######################################################################################################################
#                                                       BASE TRAINER
# ######################################################################################################################
class BaseTrainer:
    def __init__(self, cfg, logger = None):
        """
        This is a utility class that allows one to initialize in a structured way the whole training procedure.

        :param cfg:     yacs.config dict
        :param logger:  logging.logger
        """
        self.cfg = cfg
        self.logger = logger

        self.nameModels = []
        self.nameLosses = []
        self.nameOptims = []

        for init in ['init_train_data', 'init_networks', 'init_losses', 'init_optimizers']:
            getattr(self, init)(cfg)

    # =============================================== REGISTER MODEL ===================================================
    def init_train_data(self, cfg):
        raise NotImplementedError('Please initialize the Datasets/Dataloaders here.')

    # =============================================== REGISTER MODEL ===================================================
    def init_networks(self, cfg):
        raise NotImplementedError('Please initialize the networks and their according weights here.')

    # =============================================== REGISTER LOSS ====================================================
    def init_losses(self, cfg):
        raise NotImplementedError('Please initialize the losses/training criterions here and other utilities.')

    # =============================================== REGISTER OPTIMIZER ===============================================
    def init_optimizers(self, cfg):
        raise NotImplementedError('Please initialize the optimizers and schedulers here.')

    # ================================================== OPTIMIZE PARAMS ===============================================
    def optimize_parameters(self, *args, **kwargs):
        raise NotImplementedError('Please implement the training logic here.')

    # =============================================== SAVE MODELS ======================================================
    def save_models(self, saveIdx):
        """
        Save the current registered modules in the /networks folder. If the saveDir does no exists, one will be created.

        :param saveIdx: int attached to the end of the module save name (usually the step/epoch number)
        """

        if self.logger is not None:
            self.logger.debug(f'Saving epoch {saveIdx}')

        for name in self.nameModels:
            saveName = os.path.join(self.cfg.general.saveDir, 'networks')

            if not os.path.exists(saveName):
                os.makedirs(saveName)

            saveFile = os.path.join(saveName, '%s_%s.pth' % (saveIdx, name))
            model    = getattr(self, name)

            torch.save(model.cpu().state_dict(), saveFile)
            model.to(self.cfg.general.device)

        if self.logger is not None:
            self.logger.debug(f'Epoch saved [{saveIdx}]')

    # =============================================== LOAD NETWORKS ====================================================
    def load_models(self, saveIdx):
        """
        Load the networks from the configured saveDir.

        :param saveIdx: the index to load
        """

        if self.logger is not None:
            self.logger.debug(f'Loading epoch [{saveIdx}]...')

        for name in self.nameModels:
            loadName = os.path.join(self.cfg.general.saveDir, 'networks')
            loadFile = os.path.join(loadName, '%s_%s.pth' % (saveIdx, name))

            stateDict = torch.load(loadFile)

            if hasattr(stateDict, '_metadata'):
                del stateDict._metadata

            model = getattr(self, name)
            model.load_state_dict(stateDict)

        self.logger.debug(f'Epoch loaded [{saveIdx}]')

    # =============================================== SAVE OPTIMIZERS ==================================================
    def save_optimizers(self, saveIdx):
        """
        Save the current registered optimizers in the /optimizers folder. If the saveDir does no exists, one will be
        created. This is done because the optimizers have an internal state on their own, an resuming without loading
        will hurt the training.

        :param saveIdx: int attached to the end of the module save name (usually the step/epoch number)
        """
        for name in self.nameOptims:
            saveName = os.path.join(self.cfg.general.saveDir, 'optimizers')

            if not os.path.exists(saveName):
                os.makedirs(saveName)

            saveFile = os.path.join(saveName, '%s_%s.pth' % (saveIdx, name))

            optim = getattr(self, name)
            torch.save(optim.state_dict(), saveFile)

    # =============================================== LOAD OPTIMIZERS ==================================================
    def load_optimizers(self, saveIdx):
        """
        Load the registered optimizers from /optimizers folder. This is done because the optimizers have an internal
        state on their own, an resuming without loading will hurt the training.

        :param saveIdx: the index to load
        """

        for name in self.nameOptims:
            loadName = os.path.join(self.cfg.general.saveDir, 'optimizers')
            loadFile = os.path.join(loadName, '%s_%s.pth' % (saveIdx, name))

            stateDict = torch.load(loadFile)

            if hasattr(stateDict, '_metadata'):
                del stateDict._metadata

            optim = getattr(self, name)
            optim.load_state_dict(stateDict)
