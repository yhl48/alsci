from typing import Union, Optional, Sequence, List
import torch
import numpy as np
import pytorch_lightning as pl
from models.dense01 import DENSE01  # type: ignore
from algs.random_sample import RandomSample  # type: ignore
from algs.coreset_sample import CoreSetSample  # type: ignore
from pytorch_lightning.loggers import TensorBoardLogger
from models.litmodel import LitModel, DENSEDataModule  # type: ignore
from tqdm import tqdm  # type: ignore
import configparser
import h5py  # type: ignore

parser = configparser.ConfigParser()


def random_sampling(ninps: int,
                    nout: int,
                    outshape: Sequence[int],
                    total_pool_size: int,
                    jobids: np.ndarray,
                    budget: int,
                    warm_start: bool = False,
                    shrink_perturb: bool = False,
                    shrink_by: Optional[float] = 0.5,
                    perturb_by: Optional[float] = 0.1):
    """
    :param ninps: number of input parameters of the emulator
    :param nout: number of outputs of the emulator, i.e. the 'depth'
    :param outshape: number of dimensions of the emulator output, e.g. 1D or 2D
    :param nb_val: size of validation data
    :param total_pool_size: size of the total pool of labelled and unlabelled samples
    :param jobids: list of job ids
    :param model: simulator
    :param budget: number of datapoints to be selected, same for all iterations
    :param vis_depth: the output 'depth' to visualise (not actually used)
    :param warm_start: if True, the emulator is warm started in new iterations
    :param shrink_perturb: if True (and warm_start is also set to True),
                           shrink and perturb trick is used in warm start
    :param shrink_by: if shrink_perturb is set to True, the parameters (weights) are
                      shrunk by a the fraction given here
    :param perturb_by: if shrink_perturb is set to True, the parameters (weights) are
                       perturbed by Gaussian noise with standard deviation given here
    """
    parser.read('./config/config.ini')

    h5f = h5py.File('./data/shuffled_data.h5', 'r')
    x: np.ndarray = h5f['train_input'][:]
    x_val: np.ndarray = h5f['val_input'][:]
    h5f.close()

    x_train_idx: np.ndarray = np.array([])
    if warm_start:
        emulator: torch.nn.Module = DENSE01(ninps=ninps, nout=nout, outshape=outshape)
    for jobid in jobids:
        if not warm_start:
            # retrain the emulator from scratch in each jobid
            emulator = DENSE01(ninps=ninps, nout=nout, outshape=outshape)
        random_sampling: RandomSample = RandomSample(nsamples=budget,
                                                     ndim=ninps,
                                                     total_pool_size=total_pool_size,
                                                     selected_x=x_train_idx)

        # sampling (ask is the acquisition function)
        x_train_idx = random_sampling.ask_normalized(jobid=jobid)
        x_train: np.ndarray = x[x_train_idx, ...]

        # train the emulator, return the model so that output can be tested
        logger: TensorBoardLogger = TensorBoardLogger(save_dir=parser['hparams']['log_dir'],
                                                      version=parser['hparams']['rs_log_name'] + str(jobid + 1),
                                                      name=parser['hparams']['sim_model'],
                                                      default_hp_metric=False)
        emulator = random_sampling.tell_normalized(emulator=emulator,
                                                   x_train=x_train,
                                                   x_val=x_val,
                                                   jobid=jobid,
                                                   logger_name=logger)

        if warm_start and shrink_perturb:
            emulator = _shrink_perturb(emulator, shrink_by, perturb_by)


def coreset_sampling(total_pool_size: int,
                     ninps: int,
                     nout: int,
                     outshape: Sequence[int],
                     jobids: np.ndarray,
                     budget: int = 500,
                     warm_start: bool = False,
                     shrink_perturb: bool = False,
                     shrink_by: Optional[float] = 0.5,
                     perturb_by: Optional[float] = 0.1):
    """
    :param total_pool_size: size of the total pool of labelled and unlabelled samples
    :param ninps: number of input parameters of the emulator
    :param nout: number of outputs of the emulator, i.e. the 'depth'
    :param outshape: number of dimensions of the emulator output, e.g. 1D or 2D
    :param nb_val: size of validation data
    :param jobids: list of job ids
    :param model: simulator
    :param budget: size of new batch of samples in each iteration, set init_pool_size=budget here
    :param vis_depth: the output 'depth' to visualise (not actually used)
    :param warm_start: if True, the emulator is warm started in new iterations
    :param shrink_perturb: if True (and warm_start is also set to True),
                           shrink and perturb trick is used in warm start
    :param shrink_by: if shrink_perturb is set to True, the parameters (weights) are
                      shrunk by a the fraction given here
    :param perturb_by: if shrink_perturb is set to True, the parameters (weights) are
                       perturbed by Gaussian noise with standard deviation given here
    """
    parser.read('./config/config.ini')

    h5f = h5py.File('./data/shuffled_data.h5', 'r')
    x: np.ndarray = h5f['train_input'][:]
    x_val: np.ndarray = h5f['val_input'][:]
    h5f.close()

    # retrain the emulator from scratch in each jobid
    emulator: torch.nn.Module = DENSE01(ninps=ninps, nout=nout, outshape=outshape)
    # extract x_embedding from emulator
    x_embedding = get_latent_embedding(emulator, x)

    x_train_idx: Optional[np.ndarray] = None

    for jobid in jobids:
        # sampling (ask is the acquisition function)
        coreset_sampling = CoreSetSample(x_embedding, budget=budget, t_idx=x_train_idx, init_pool_size=budget)
        # x_train is the input, not the middle layer representation
        x_train_idx = coreset_sampling.ask_normalized(jobid=jobid)
        x_train: np.ndarray = x[x_train_idx, ...]

        # train the emulator from scratch, return the model so that output can be tested
        logger: TensorBoardLogger = TensorBoardLogger(save_dir=parser['hparams']['log_dir'],
                                                      version=parser['hparams']['cs_log_name'] + str(jobid + 1),
                                                      name=parser['hparams']['sim_model'],
                                                      default_hp_metric=False)
        emulator = coreset_sampling.tell_normalized(emulator=emulator,
                                                    x_train=x_train,
                                                    x_val=x_val,
                                                    jobid=jobid,
                                                    logger_name=logger)

        if jobid < jobids[-1]:
            x_embedding = get_latent_embedding(emulator, x)
            # create a new emulator instance to override the old one to retrain from scratch in the next iteration
            if not warm_start:
                emulator = DENSE01(ninps=ninps, nout=nout, outshape=outshape)
            else:
                if shrink_perturb:
                    emulator = _shrink_perturb(emulator, shrink_by, perturb_by)


def _shrink_perturb(emulator, shrink_scale, noise_scale):
    with torch.no_grad():
        for param in emulator.parameters():
            param.add_(-param * shrink_scale)
            param.add_(torch.randn(param.size()) * noise_scale)
    return emulator

def get_latent_embedding(emulator: torch.nn.Module, x_all: np.ndarray, avg=True):
    # extract embeddings from emulator, could be final layer output depending on emulator forward
    # in LitModel
    parser.read('./config/config.ini')
    data_module = DENSEDataModule(x_pred=x_all,
                                  inference_batch_size=int(parser['hparams']['inference_batch_size']),
                                  predict=True)
    dense_model = LitModel(emulator)
    trainer = pl.Trainer(gpus=int(parser['hparams']['nb_gpus']), logger=False, accelerator='ddp')

    x_embedding: List = []
    for _ in tqdm(range(int(parser['hparams']['npass']))):
        out = trainer.predict(model=dense_model, datamodule=data_module, return_predictions=True)
        assert out is not None
        out = [item for sublist in out for item in sublist]
        x_embedding.append(torch.Tensor(out))
    res: np.ndarray = torch.stack([*x_embedding], dim=0).detach().numpy()
    if avg is False:
        return res
    else:
        res = np.mean(res, axis=0)
        return res
