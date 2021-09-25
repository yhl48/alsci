"""
Partially referred to (https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py)
with a substantial modification

Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance. Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from typing import Tuple, Optional, List
from tqdm import tqdm  # type: ignore
from pykeops.torch import Vi, Vj  # type: ignore


class kCenterGreedy():
    def __init__(self, x: np.ndarray, cluster_centers: Optional[list] = None):
        """
        :param x: the features of the pool of ALL samples from which smaller subset will be sampled
                  to train the deep learning model (features normally taken from hidden layer)
        :param cluster_centers: indices of the current pool of labelled training samples
        """
        self.name: str = 'kcenter'
        self.n: int = x.shape[0]
        self.x: np.ndarray = x.reshape(x.shape[0], -1)
        self.min_distances: Optional[np.ndarray] = None
        self.cluster_centers: Optional[list] = cluster_centers  # indices of cluster centers, not coordinates
        use_cuda = torch.cuda.is_available()
        self.tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  # type: ignore

    def update_distances(self, initial_state: bool = False):
        center: np.ndarray
        dist: np.ndarray

        if initial_state:
            center = self.x[self.cluster_centers, :]
            x = torch.as_tensor(self.x).type(self.tensor)
            center = torch.as_tensor(center).type(self.tensor)
            x_i = Vi(x)
            y_j = Vj(center)
            dist = ((x_i - y_j) ** 2).sum().min(dim=1).detach().cpu().numpy()
            self.min_distances = dist.reshape(-1, 1)
            return self.min_distances

        else:
            # add this if statement so that it is mypy compliant
            if self.cluster_centers is not None:
                center = np.array([self.x[self.cluster_centers[-1], :]])
                x = torch.as_tensor(self.x).type(self.tensor)
                center = torch.as_tensor(center).type(self.tensor)
                x_i = Vi(x)
                y_j = Vj(center)
                dist = ((x_i - y_j) ** 2).sum().min(dim=1).detach().cpu().numpy()
                # add this if statement so that it is mypy compliant
                if self.min_distances is not None:
                    assert self.min_distances.shape == dist.shape
                    self.min_distances = np.minimum(self.min_distances, dist)
                    # assert self.min_distances.shape == (self.n, 1)

    def select_batch(self, b: int, initial_pool_size: Optional[int] = None) -> Tuple[List, List, List]:
        """
        :param b: number of datapoints to be selected
        :param init_pool_size: |s0| per the core-set paper, the initial number of cluster centers i.e. training
                               datapoints
        """
        s_next: list = []
        s0: list = []
        idx: np.ndarray

        if self.cluster_centers is None:
            idx = np.arange(initial_pool_size)
            s0 = idx.tolist()
            self.cluster_centers = s0
            self.update_distances(initial_state=True)
            s_next = s0
            return s_next, s0, self.cluster_centers

        self.update_distances(initial_state=True)
        s0 = self.cluster_centers.copy()
        for i in tqdm(range(b)):
            # add this if statement so that it is mypy compliant
            if self.min_distances is not None:
                idx = np.array(np.argmax(self.min_distances))
                self.cluster_centers.append(idx.tolist())
                self.update_distances()
                s_next.append(idx.tolist())

        return s_next, s0, self.cluster_centers
