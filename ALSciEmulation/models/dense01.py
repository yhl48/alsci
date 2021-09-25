from typing import List, Tuple, Optional, Sequence, Union, Callable
import numpy as np
import torch
import configparser

parser = configparser.ConfigParser()

__all__ = ["DENSE01"]


class DENSE01(torch.nn.Module):
    def __init__(self, ninps: int = 3, nout: int = 1, outshape: Sequence[int] = [250],
                 kernel_size: int = 3):
        super().__init__()
        """
        ndim=len(outshape) should be output type per the paper, for e.g. 2D for images, 1D for lines
        nout should be the 'number of columns', or 'number of channels' in the final output
        ndepths should be the number of CNN layers
        """
        parser.read('/home/yi_heng_machine_discovery_com/aldense/aldense/config/config.ini')
        # determine the channels and shapes of the intermediate values
        ch = max(64, nout)
        ndim = len(outshape)  # number of dimension of the output signal
        ndepths = 7
        if ndim > 0:
            shapes: List[Sequence[int]] = list(zip(*[generate_shape(outs, ndepths) for outs in outshape]))
            outsh = outshape
        else:
            shapes = [(1,)] * ndepths
            outsh = [1]
        nodes: List[Tuple[int, ...]] = []
        for shape in shapes:
            nodes.append((ch, *shape))
        # print(nodes)
        nodes.append((nout, *outsh))
        # print(nodes)
        self._nodes = nodes
        self._ndim = ndim
        """
        # densely connected layer
        self._dense = torch.nn.Sequential(
            torch.nn.Linear(ninps, ch * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(ch * 4, ch * np.prod(nodes[0][1:])),
        )
        """
        # densely connected layer
        self._dense = torch.nn.Sequential(
            torch.nn.Linear(ninps, ch * 4),
            torch.nn.ReLU(),
            # torch.nn.ELU(),
            torch.nn.Dropout(p=float(parser['hparams']['dropout'])),
            # torch.nn.BatchNorm1d(ch * 4),
            torch.nn.Linear(ch * 4, ch * np.prod(nodes[0][1:])),
        )

        # construct the convolutional layer and skip connection
        convs: List[torch.nn.Module] = []
        skips: List[torch.nn.Module] = []
        for i in range(len(nodes) - 1):
            ch0 = nodes[i][0]
            ch1 = nodes[i + 1][0]
            sz0 = nodes[i][1:]
            sz1 = nodes[i + 1][1:]

            with_relu = i < len(nodes) - 1
            if with_relu:
                nonlinear: torch.nn.Module = torch.nn.Sequential(torch.nn.ReLU(), DCTHybrid())
                # nonlinear: torch.nn.Module = torch.nn.Sequential(torch.nn.ELU(), DCTHybrid())
            else:
                nonlinear = DCTHybrid()

            padding = (kernel_size - 1) // 2
            if ndim == 1 or ndim == 0:
                # convolutional layer
                conv = torch.nn.Sequential(
                    ConvertSize1D(sz0, sz1, fill_const=True),
                    torch.nn.Conv1d(ch0, ch1, kernel_size=kernel_size,
                                    padding=padding, bias=False),
                    nonlinear,
                )
                # skip connection
                skip = torch.nn.Sequential(
                    ConvertSize1D(sz0, sz1),
                    ConvertChannel(ch0, ch1, max(ndim, 1))
                )
            elif ndim == 2:
                # convolutional layer
                conv = torch.nn.Sequential(
                    ConvertSize2D(sz0, sz1, fill_const=True),
                    torch.nn.Conv2d(ch0, ch1, kernel_size=kernel_size,
                                    padding=padding, bias=False),
                    nonlinear,
                )
                # skip connection
                skip = torch.nn.Sequential(
                    ConvertSize2D(sz0, sz1),
                    ConvertChannel(ch0, ch1, max(ndim, 1))
                )
            else:
                raise NotImplementedError("Not implemented operation for ndim: %d" % ndim)

            convs.append(conv)
            skips.append(skip)

        self._convs = torch.nn.ModuleList(convs)
        self._skips = torch.nn.ModuleList(skips)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(self._convs) == len(self._skips)
        x = self._dense(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], *self._nodes[0])
        for conv, skip in zip(self._convs, self._skips):
            x = conv(x) + skip(x)
            # print(x.shape)
        # x is not (nbatch, nout, *outshape)
        if self._ndim == 0:
            x = x.reshape(*x.shape[:2])
        # for name, param in self.named_parameters():
        #     print(name, param.shape)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dense(x)
        """x = x.reshape(x.shape[0], *self._nodes[0])
        for conv, skip in zip(self._convs[:2], self._skips[:2]):
            x = conv(x) + skip(x)
            # print(x.shape)
        x = torch.mean(x, dim=1)"""
        # print(x.shape)
        return x


class ConvertSize1D(torch.nn.Module):
    def __init__(self, s0: Tuple[int, ...], s1: Tuple[int, ...], fill_const: bool = False):
        super().__init__()
        assert len(s0) == len(s1) == 1

        if s0 == s1:
            self._module: torch.nn.Module = torch.nn.Identity()

        else:
            ratio = int(np.round(s1[0] * 1.0 / s0[0]))
            shrink = ratio * s0[0] - s1[0]
            if shrink == 0:
                self._module = Upsample1D(ratio, fill_const)
            else:
                self._module = torch.nn.Sequential(
                    Upsample1D(ratio, fill_const),
                    Pad1D(shrink // 2, shrink - shrink // 2)
                )

    def forward(self, x):
        return self._module(x)


class Upsample1D(torch.nn.Module):
    def __init__(self, nr: int = 2, fill_const: bool = False):
        super().__init__()
        self.nr = nr
        self.repeats = (1, 1, 1, nr)
        self.fill_const = fill_const
        if fill_const:
            self.const = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        yshape = (x.shape[0], x.shape[1], -1)
        if not self.fill_const:
            return x.unsqueeze(-1).repeat(*self.repeats).view(yshape)
        else:
            x = x - self.const
            return torch.nn.functional.pad(x.unsqueeze(-1), (0, self.nr - 1)).view(yshape) + self.const


class Pad1D(torch.nn.Module):
    def __init__(self, nl: int = 1, nr: int = 1):
        super().__init__()

        if nl + nr == 0:
            self.module: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.Identity()
        else:
            self.module = lambda x: x[:, :, nl:-nr]

    def forward(self, x):
        return self.module(x)


class ConvertSize2D(torch.nn.Module):
    def __init__(self, s0: Tuple[int, ...], s1: Tuple[int, ...], fill_const: bool = False):
        super().__init__()

        if s0 == s1:
            self.module: torch.nn.Module = torch.nn.Identity()

        else:
            ratio = [int(np.round(s1[i] * 1.0 / s0[i])) for i in range(len(s0))]
            shrink = [ratio[i] * s0[i] - s1[i] for i in range(len(s0))]
            if shrink[0] == 0 and shrink[1] == 0:
                self.module = Upsample2D(ratio[0], ratio[1], fill_const)
            else:
                self.module = torch.nn.Sequential(
                    Upsample2D(ratio[0], ratio[1], fill_const),
                    Pad2D(shrink[0] // 2, shrink[0] - shrink[0] // 2,
                          shrink[1] // 2, shrink[1] - shrink[1] // 2)
                )

    def forward(self, x):
        return self.module(x)


class Upsample2D(torch.nn.Module):
    def __init__(self, nr1: int = 2, nr2: int = 2, fill_const: bool = False):
        super().__init__()
        self.nr1 = nr1
        self.nr2 = nr2
        self.repeats2 = (1, 1, 1, 1, nr2)
        self.repeats1 = (1, 1, 1, nr1, 1)
        self.repeats = (1, 1, 1, nr1, 1, nr2)
        self.pads = (0, self.nr2 - 1, 0, 0, 0, self.nr1 - 1)
        self.fill_const = fill_const
        if fill_const:
            self.const = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yshape_horz = (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * self.nr2)
        yshape_out = (x.shape[0], x.shape[1], x.shape[2] * self.nr1, x.shape[3] * self.nr2)
        if not self.fill_const:
            return x.unsqueeze(-2).unsqueeze(-1).repeat(*self.repeats).view(yshape_out)
        else:
            x = x - self.const
            y = torch.nn.functional.pad(x.unsqueeze(-2).unsqueeze(-1), self.pads).view(yshape_out)
            return y + self.const


class Pad2D(torch.nn.Module):
    def __init__(self, nl1: int = 1, nr1: int = 1, nl2: int = 1, nr2: int = 1):
        super().__init__()

        if nl1 == 0 and nl2 == 0 and nr1 == 0 and nr2 == 0:
            self.module: Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.Identity()
        elif nl1 == 0 and nr1 == 0:
            self.module = lambda x: x[:, :, :, nl2:-nr2]
        elif nl2 == 0 and nr2 == 0:
            self.module = lambda x: x[:, :, nl1:-nr1, :]
        else:
            self.module = lambda x: x[:, :, nl1:-nr1, nl2:-nr2]

    def forward(self, x):
        return self.module(x)


class ConvertChannel(torch.nn.Module):
    def __init__(self, c0, c1, ndim):
        super().__init__()

        if c0 == c1:
            self.module = torch.nn.Identity()
        elif c0 > c1:
            if ndim == 1:
                self.module = lambda x: x[:, :c1, :]
            elif ndim == 2:
                self.module = lambda x: x[:, :c1, :, :]
            elif ndim == 3:
                self.module = lambda x: x[:, :c1, :, :, :]
            else:
                raise RuntimeError("Invalid ndim: %d" % ndim)
        else:
            raise NotImplementedError("Upping the channels is not implemented yet")

    def forward(self, x):
        return self.module(x)


def dct(x: torch.Tensor, norm: Optional[str] = None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=-1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


class DCTHybrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._alpha = torch.nn.Parameter(torch.tensor(1.0))
        self._beta = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = dct(x, norm="ortho")
        return self._alpha * x + self._beta * xd


def generate_shape(s: int, n: int) -> List[int]:
    # generate the shape for the intermediate values
    s = int(s)
    # get the lowest 2^p not lower than s
    p = int(np.ceil(np.log2(s)))
    p2 = 2**p
    sp = p2 - s
    binsp = bin(sp)[2:]
    if len(binsp) < p:
        binsp = "0" * (p - len(binsp)) + binsp
    binsp_list = [int(c) for c in binsp]
    res = [1]
    for i in range(len(binsp_list)):
        r = res[-1] * 2
        if binsp_list[i]:
            r = r - 1
        res.append(r)
    res.pop(0)
    if len(res) >= n:
        res = res[-n:]
    else:
        res = [res[0]] * (n - len(res)) + res
    return res


if __name__ == "__main__":
    outshape = [250]
    nout = 2
    ninps = 5
    # nn0d = DENSE01(ninps=3, nout=2, outshape=())
    # nn1d = DENSE01(ninps=3, nout=2, outshape=(250,))
    # nn2d = DENSE01(ninps=3, nout=2, outshape=(126, 128))
    nn1d = DENSE01(ninps=ninps, nout=nout, outshape=outshape)
    x = torch.rand((1, ninps))
    nn1d(x)
    # print(nn1d.get_embeddings(x).shape)
    # nn1d.get_embeddings(x)
    # print(nn0d(x).shape)
    # print(nn1d(x).shape)
    # print(nn2d(x).shape)
