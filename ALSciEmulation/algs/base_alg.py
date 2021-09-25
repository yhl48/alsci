from abc import abstractmethod
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger

class BaseAlg(object):
    @abstractmethod
    def ask_normalized(self, jobid: int) -> np.ndarray:
        """
        Given the job id, returns an ndarray with shape of ``(nsamples, ndim)``
        correspond to the points to be evaluated in simulations.
        The values of the output should be bounded from 0 to 1.
        """
        pass

    # @abstractmethod
    # def tell_normalized(self, x: np.ndarray, y: np.ndarray, jobid: int) -> None:
    #     """
    #     Inform the algorithm about the simulation results where ``x`` is the evaluted
    #     points and ``y`` is the evaluated outputs.
    #     """
    #     pass

    @abstractmethod
    def tell_normalized(self,
                        emulator: torch.nn.Module,
                        x_train: np.ndarray,
                        x_val: np.ndarray,
                        jobid: int,
                        logger_name: TensorBoardLogger) -> torch.nn.Module:
        """
        Inform the algorithm about the simulation results where ``x`` is the evaluted
        points and ``y`` is the evaluated outputs.
        """
        pass
