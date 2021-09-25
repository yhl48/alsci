import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import configparser
from typing import Optional
import h5py  # type: ignore
import numpy_indexed as npi  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


class LitModel(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.parser = configparser.ConfigParser()
        self.parser.read('./config/config.ini')
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def huber_loss(self, y_hat, y):
        loss = torch.nn.L1Loss(reduction='mean')
        return loss(y_hat, y)

    def on_train_start(self):
        hparams = {'hp/lr': float(self.parser['hparams']['lr']),
                   'hp/nb_epoch': int(self.parser['hparams']['nb_epoch']),
                   'hp/nb_gpus': int(self.parser['hparams']['nb_gpus']),
                   'hp/batch_size': int(self.parser['hparams']['batch_size']),
                   'hp/inference_batch_size': int(self.parser['hparams']['inference_batch_size']),
                   'hp/total_pool_size': int(self.parser['hparams']['total_pool_size']),
                   'hp/budget': int(self.parser['hparams']['budget']),
                   'hp/init_train_seed': int(self.parser['hparams']['init_train_seed']),
                   'hp/early_stopping_patience': int(self.parser['hparams']['early_stopping_patience']),
                   'hp/gradient_clip_val': float(self.parser['hparams']['gradient_clip_val']),
                   'hp/shrink_by': float(self.parser['hparams']['shrink_by']),
                   'hp/perturb_by': float(self.parser['hparams']['perturb_by']),
                   'hp/check_val_every_n_epoch': int(self.parser['hparams']['check_val_every_n_epoch'])}

        self.logger.log_hyperparams(self.hparams, hparams)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.backbone.train()  # temporary solution to enable dropout at prediction
        x = batch
        y = self.backbone(x)
        y = self.all_gather(y)
        gpu_n, batch_n, out_n, outshape_n = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = torch.cat([y[i, ...] for i in range(y.shape[0])], dim=-1).reshape(-1, y.shape[-1])
        y = y.reshape(y.shape[0], out_n, outshape_n)
        # need to store in cpu otherwise CUDA OOM
        return y.detach().cpu().tolist()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.huber_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss': loss, 'y': y, 'y_hat': y_hat}

    def training_epoch_end(self, outputs):
        if self.current_epoch % int(self.parser['hparams']['log_plot']) == 0:
            loss, y, y_hat = outputs[-1]['loss'], outputs[-1]['y'], outputs[-1]['y_hat']

            y = self.all_gather(y)
            gpu_n, batch_n, out_n, outshape_n = y.shape
            y = y.reshape(y.shape[0], y.shape[1], -1)
            y = torch.cat([y[i, ...] for i in range(y.shape[0])], dim=-1).reshape(-1, y.shape[-1])
            y = y.reshape(y.shape[0], out_n, outshape_n)

            y_hat = self.all_gather(y_hat)
            gpu_n, batch_n, out_n, outshape_n = y_hat.shape
            y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], -1)
            y_hat = torch.cat([y_hat[i, ...] for i in range(y_hat.shape[0])], dim=-1).reshape(-1, y_hat.shape[-1])
            y_hat = y_hat.reshape(y_hat.shape[0], out_n, outshape_n)

            img = y_hat[-1, ...].detach().cpu().numpy()
            true_img = y[-1, ...].detach().cpu().numpy()
            figs = []
            for i in range(img.shape[0]):
                fig = plt.figure()
                plt.plot(img[i], label='prediction')
                plt.plot(true_img[i], label='truth')
                plt.legend()
                plt.close()
                figs.append(fig)
            self.logger.experiment.add_figure('training sample 1', figs, self.current_epoch)

            img = y_hat[-2, ...].detach().cpu().numpy()
            true_img = y[-2, ...].detach().cpu().numpy()
            figs = []
            for i in range(img.shape[0]):
                fig = plt.figure()
                plt.plot(img[i], label='prediction')
                plt.plot(true_img[i], label='truth')
                plt.legend()
                plt.close()
                figs.append(fig)
            self.logger.experiment.add_figure('training sample 2', figs, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.huber_loss(y_hat, y)

        self.log('step', self.trainer.current_epoch, sync_dist=True)  # change x-axis from step to epoch
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, y, y_hat

    def validation_epoch_end(self, outputs):
        if self.current_epoch % int(self.parser['hparams']['log_plot']) == 0:
            loss, y, y_hat = outputs[-1]

            y = self.all_gather(y)
            gpu_n, batch_n, out_n, outshape_n = y.shape
            y = y.reshape(y.shape[0], y.shape[1], -1)
            y = torch.cat([y[i, ...] for i in range(y.shape[0])], dim=-1).reshape(-1, y.shape[-1])
            y = y.reshape(y.shape[0], out_n, outshape_n)

            y_hat = self.all_gather(y_hat)
            gpu_n, batch_n, out_n, outshape_n = y_hat.shape
            y_hat = y_hat.reshape(y_hat.shape[0], y_hat.shape[1], -1)
            y_hat = torch.cat([y_hat[i, ...] for i in range(y_hat.shape[0])], dim=-1).reshape(-1, y_hat.shape[-1])
            y_hat = y_hat.reshape(y_hat.shape[0], out_n, outshape_n)

            img = y_hat[-1, ...].detach().cpu().numpy()
            true_img = y[-1, ...].detach().cpu().numpy()
            figs = []
            for i in range(img.shape[0]):
                fig = plt.figure()
                plt.plot(img[i], label='prediction')
                plt.plot(true_img[i], label='truth')
                plt.legend()
                plt.close()
                figs.append(fig)
            self.logger.experiment.add_figure('val sample 1', figs, self.current_epoch)

            img = y_hat[-2, ...].detach().cpu().numpy()
            true_img = y[-2, ...].detach().cpu().numpy()
            figs = []
            for i in range(img.shape[0]):
                fig = plt.figure()
                plt.plot(img[i], label='prediction')
                plt.plot(true_img[i], label='truth')
                plt.legend()
                plt.close()
                figs.append(fig)
            self.logger.experiment.add_figure('val sample 2', figs, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.huber_loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(self.parser['hparams']['lr']), amsgrad=True)
        return optimizer


class DENSEDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None, labelling: bool = False):
        self.x: torch.Tensor = torch.Tensor(x)
        self.labelling: bool = labelling
        if self.labelling is False:
            self.y: torch.Tensor = torch.Tensor(y)
            assert len(self.x) == len(self.y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        if self.labelling is False:
            return self.x[index], self.y[index]
        else:
            return self.x[index]


class DENSEDataModule(pl.LightningDataModule):
    def __init__(self,
                 x_train: Optional[np.ndarray] = None,
                 x_val: Optional[np.ndarray] = None,
                 x_pred: Optional[np.ndarray] = None,
                 batch_size: int = 32,
                 inference_batch_size: int = 64,
                 predict: bool = False,
                 jobid: Optional[int] = None):
        super().__init__()

        if predict is False:
            self.batch_size: int = batch_size
            assert x_train is not None
            self.x_train: DENSEDataset = DENSEDataset(x_train, labelling=True)
            assert x_val is not None
            self.x_val: DENSEDataset = DENSEDataset(x_val, labelling=True)

            h5f = h5py.File('./data/shuffled_data.h5', 'r')
            input_train: np.ndarray = h5f['train_input'][:]
            output_train: np.ndarray = h5f['train_output'][:]
            y_val: np.ndarray = h5f['val_output'][:]
            h5f.close()

            y_train = output_train[np.flatnonzero(npi.contains(x_train, input_train))]
            x_train = input_train[np.flatnonzero(npi.contains(x_train, input_train))]

            assert x_train is not None and y_train is not None
            self.train_data: DENSEDataset = DENSEDataset(x_train, y_train, labelling=False)
            assert x_val is not None
            self.val_data: DENSEDataset = DENSEDataset(x_val, y_val, labelling=False)

        else:
            self.inference_batch_size: int = inference_batch_size
            assert x_pred is not None
            self.pred_data: DENSEDataset = DENSEDataset(x_pred, labelling=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        # return DataLoader(self.val_data, batch_size=self.batch_size)
        return DataLoader(self.val_data, batch_size=3000)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.pred_data, batch_size=self.inference_batch_size)
