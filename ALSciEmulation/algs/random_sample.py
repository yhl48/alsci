from algs.base_alg import BaseAlg  # type: ignore
from models.litmodel import LitModel, DENSEDataModule  # type: ignore
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Union
import configparser


class RandomSample(BaseAlg):
    def __init__(self,
                 nsamples: int,
                 ndim: int,
                 total_pool_size: int,
                 selected_x: np.ndarray = np.array([])):
        """
        :param nsamples: number of datapoints to be selected
        :param ndim: not used here (remove later)
        :param total_pool_size: size of the total pool of labelled and unlabelled samples
        :param selected_x: indices of the current pool of labelled training samples
        """
        self._ndim: int = ndim
        self._nsamples: int = nsamples
        self.selected_x: np.ndarray = selected_x  # the index
        self.total_pool_size: int = total_pool_size

        self.parser = configparser.ConfigParser()
        self.parser.read('/home/yi_heng_machine_discovery_com/aldense/aldense/config/config.ini')
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
        :param jobid: jobid

        :return: the indices of all the current pool of (& to-be) labelled training samples (from all iterations)
        """
        if jobid == 0:
            self.selected_x = np.arange(self._nsamples)
            return self.selected_x
        else:
            supposed_n = self.selected_x.shape[0] + self._nsamples
            while self.selected_x.shape[0] < supposed_n:
                choice = np.random.choice(self.total_pool_size)
                if choice not in self.selected_x:
                    self.selected_x = np.append(self.selected_x, np.array(choice))
            return self.selected_x

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
