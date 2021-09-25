from algs.base_alg import BaseAlg  # type: ignore
from algs.k_center_greedy import kCenterGreedy  # type: ignore
from models.litmodel import LitModel, DENSEDataModule  # type: ignore
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional, Union
import configparser


class CoreSetSample(BaseAlg):
    def __init__(self, x: np.ndarray, budget: int, t_idx: Optional[np.ndarray] = None,
                 init_pool_size: Optional[int] = None):
        """
        Implementation of the paper https://arxiv.org/abs/1708.00489

        :param x: the features of the pool of ALL samples from which smaller subset will be sampled
                  to train the deep learning model (features normally taken from hidden layer)
        :param budget: number of datapoints to be selected
        :param t_idx: indices of the current pool of labelled training samples
        :param init_pool_size: |s0| per the core-set paper, the initial number of cluster centers i.e. training
                               datapoints

        if t_idx is None, init_pool_size must be provided
        if t_idx is not None, init_pool_size can be omitted
        & vice versa
        """
        self.x: np.ndarray = x
        self.t_idx: Optional[list] = t_idx.tolist() if t_idx is not None else None  # indices of current pool
        self.algo = kCenterGreedy(x=self.x, cluster_centers=self.t_idx)  # self.t_idx can be a list or None
        self.s0: Optional[int] = init_pool_size  # number of initial cluster centers to be chosen if no t_idx given
        self.budget: int = budget

        self.parser = configparser.ConfigParser()
        self.parser.read('./config/config.ini')
        self.patience: int = int(self.parser['hparams']['early_stopping_patience'])
        self.gradient_clip_val: float = float(self.parser['hparams']['gradient_clip_val'])
        self.track_grad_norm: Union[int, str]
        try:
            self.track_grad_norm = int(self.parser['hparams']['track_grad_norm'])
        except Exception:
            self.track_grad_norm = self.parser['hparams']['track_grad_norm']
        self.batch_size: int = int(self.parser['hparams']['batch_size'])
        self.nb_epoch: int = int(self.parser['hparams']['nb_epoch'])
        self.nb_gpus: int = int(self.parser['hparams']['nb_gpus'])
        self.check_val_every_n_epoch: int = int(self.parser['hparams']['check_val_every_n_epoch'])

    def ask_normalized(self, jobid: int) -> np.ndarray:
        """
        :param jobid: jobid (not used here, just a placeholder in case it is needed in the future)

        :return: the indices of all the current pool of (& to-be) labelled training samples (from all iterations)
        """
        # not the most efficient way, should only return the to-be samples
        return np.array(self.algo.select_batch(b=self.budget, initial_pool_size=self.s0)[2])

    def tell_normalized(self,
                        emulator: torch.nn.Module,
                        x_train: np.ndarray,
                        x_val: np.ndarray,
                        jobid: int,
                        logger_name: TensorBoardLogger) -> torch.nn.Module:
        """
        :param simulator: oracle for data labelling
        :param emulator: emulator to be trained
        :param x_train: training points x (the input, different from self.x)
        :param x_val: validation points x
        :param jobid: jobid (not used here, just a placeholder in case it is needed in the future)
        :param logger_name: Tensorboard log file details

        :return: the trained emulator
        """
        data_module = DENSEDataModule(x_train=x_train,
                                      x_val=x_val,
                                      batch_size=self.batch_size)
        dense_model = LitModel(emulator)
        trainer = pl.Trainer(max_epochs=self.nb_epoch,
                             gpus=self.nb_gpus,
                             logger=logger_name,
                             accelerator='ddp',
                             gradient_clip_val=self.gradient_clip_val,
                             track_grad_norm=self.track_grad_norm,
                             terminate_on_nan=True,
                             check_val_every_n_epoch=self.check_val_every_n_epoch,
                             callbacks=[EarlyStopping(monitor='val_loss',
                                                      patience=self.patience,
                                                      verbose=False),
                                        ModelCheckpoint(save_last=True)])
        trainer.fit(dense_model, data_module)
        return emulator
