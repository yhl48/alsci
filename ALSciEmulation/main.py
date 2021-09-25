import numpy as np
from models.train import random_sampling, coreset_sampling
import configparser
import h5py  # type: ignore
from sklearn.utils import shuffle  # type: ignore
import argparse


if __name__ == "__main__":
    """
    Allow users to set hyperparameters using command line
    """
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--sampling_method", type=str, default='coreset', help="sampling method to be used",
                           choices={"coreset", "random"})
    argparser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    argparser.add_argument("--nb_epoch", type=int, default=2000, help="number of epochs per active learning iteration")
    argparser.add_argument("--nb_gpus", type=int, default=-1, help="number of GPUs, -1 to use all GPUs available")
    argparser.add_argument("--batch_size", type=int, default=32, help="size of the mini-batches")
    argparser.add_argument("--inference_batch_size", type=int, default=32, help="batch size when making predictions")
    argparser.add_argument("--jobids", type=int, default=5, help="number of active learning iteration")
    argparser.add_argument("--budget", type=int, default=2000, help="budget per iteration")
    argparser.add_argument("--early_stopping_patience", type=int, default=2000, help="early stopping")
    argparser.add_argument("--log_dir", type=str, default='./log', help="directory to store training information")
    argparser.add_argument("--sim_model", type=str, default='xes', help="model name", choices={"halo", "xes"})
    argparser.add_argument("--track_grad_norm", type=int, default=-1, help="gradient norm to track, -1 to not track")
    argparser.add_argument("--log_plot", type=int, default=100,
                           help="number of epochs between subsequent plots in tensorboard")
    argparser.add_argument("--gradient_clip_val", type=int, default=1, help="clip the gradient by this amount")
    argparser.add_argument("--warm_start", type=bool, default=True,
                           help="warm start the optimisation after the first iteration")
    argparser.add_argument("--shrink_perturb", type=bool, default=True, help="warm start with shrink and perturb trick")
    argparser.add_argument("--shrink_by", type=float, default=0.5,
                           help="shrink the parameters by this fraction for warm start optimisation")
    argparser.add_argument("--perturb_by", type=float, default=0.1,
                           help="perturb the parameters by zero mean Gaussian noise with this standard deviation for \
                               warm start optimisation")
    argparser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="check validation every n epoch")
    hparams = argparser.parse_args()

    """
    Read the dataset selected by the user, and define the input and output dimensions based on the dataset
    """
    if hparams.sim_model == 'xes':
        h5f = h5py.File('./data/xes_sim_data.h5', 'r')
        train_input = h5f['train_input'][:]
        train_output = h5f['train_output'][:]
        val_input = h5f['val_input'][:]
        val_output = h5f['val_output'][:]
        test_input = h5f['test_input'][:]
        test_output = h5f['test_output'][:]
        h5f.close()
        outshape = [250]
        nout = 1
        ninps = 10
        total_pool_size = len(train_input)

    elif hparams.sim_model == 'halo':
        h5f = h5py.File('./data/halo_sim_data.h5', 'r')
        train_input = h5f['train_input'][:]
        train_output = h5f['train_output'][:]
        val_input = h5f['val_input'][:]
        val_output = h5f['val_output'][:]
        test_input = h5f['test_input'][:]
        test_output = h5f['test_output'][:]
        h5f.close()
        outshape = [250]
        nout = 1
        ninps = 5
        total_pool_size = len(train_input)

    """
    Copy the user-defined hyperparameters into a global config file
    """
    f = './config/config.ini'
    parser = configparser.ConfigParser()
    parser.read(f)
    parser.set('hparams', 'lr', str(hparams.lr))
    parser.set('hparams', 'nb_epoch', str(hparams.nb_epoch))
    parser.set('hparams', 'nb_gpus', str(hparams.nb_gpus))
    parser.set('hparams', 'batch_size', str(hparams.batch_size))
    parser.set('hparams', 'inference_batch_size', str(hparams.inference_batch_size))
    parser.set('hparams', 'jobids', str(hparams.jobids))
    parser.set('hparams', 'budget', str(hparams.budget))
    parser.set('hparams', 'early_stopping_patience', str(hparams.early_stopping_patience))  # no early stopping
    parser.set('hparams', 'log_dir', str(hparams.log_dir))
    parser.set('hparams', 'sim_model', str(hparams.sim_model))
    parser.set('hparams', 'log_plot', str(hparams.log_plot))
    parser.set('hparams', 'gradient_clip_val', str(hparams.gradient_clip_val))
    parser.set('hparams', 'track_grad_norm', str(hparams.track_grad_norm))
    parser.set('hparams', 'shrink_by', str(hparams.shrink_by))
    parser.set('hparams', 'perturb_by', str(hparams.perturb_by))
    parser.set('hparams', 'check_val_every_n_epoch', str(hparams.check_val_every_n_epoch))
    parser.set('hparams', 'total_pool_size', str(total_pool_size))
    parser.set('hparams', 'warm_start', str(hparams.warm_start))
    parser.set('hparams', 'shrink_perturb', str(hparams.shrink_perturb))

    """
    Run the selected experiment with 5 different seeds
    """
    np.random.seed(666)
    seeds = np.random.randint(3000, 10000, 5)
    for i in range(len(seeds)):
        # export HDF5_USE_FILE_LOCKING=FALSE
        x, y = shuffle(train_input, train_output, random_state=seeds[i])
        h5f = h5py.File('./data/shuffled_data.h5', 'w')
        h5f['train_input'] = x
        h5f['train_output'] = y
        h5f['val_input'] = val_input
        h5f['val_output'] = val_output
        h5f['test_input'] = test_input
        h5f['test_output'] = test_output
        h5f.close()

        parser.set('hparams', 'init_train_seed', str(seeds[i]))

        shrink_by = float(parser['hparams']['shrink_by'])
        perturb_by = float(parser['hparams']['perturb_by'])

        if hparams.sampling_method == 'coreset':
            name = 'coreset_' + 'v' + str(i) + '_'
            parser.set('hparams', 'cs_log_name', name)
            with open(f, 'w') as configfile:
                parser.write(configfile)
            coreset_sampling(int(parser['hparams']['total_pool_size']),
                             ninps,
                             nout,
                             outshape,
                             np.arange(int(parser['hparams']['jobids'])),
                             budget=int(parser['hparams']['budget']),
                             warm_start=bool(parser['hparams']['warm_start']),
                             shrink_perturb=bool(parser['hparams']['shrink_perturb']),
                             shrink_by=shrink_by,
                             perturb_by=perturb_by)
        elif hparams.sampling_method == 'random':
            name = 'random_' + 'v' + str(i) + '_'
            parser.set('hparams', 'rs_log_name', name)
            with open(f, 'w') as configfile:
                parser.write(configfile)
            random_sampling(ninps,
                            nout,
                            outshape,
                            int(parser['hparams']['total_pool_size']),
                            np.arange(int(parser['hparams']['jobids'])),
                            budget=int(parser['hparams']['budget']),
                            warm_start=bool(parser['hparams']['warm_start']),
                            shrink_perturb=bool(parser['hparams']['shrink_perturb']),
                            shrink_by=shrink_by,
                            perturb_by=perturb_by)
