import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super(Invertible1x1Conv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj

class ActivationNormalisation(nn.Module):
    def __init__(self, num_features, scale=1.):
        super(ActivationNormalisation, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -1 * utilities.mean_over_dimensions(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = utilities.mean_over_dimensions((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if not reverse:
            x = x * logs.exp()
        else:
            x = x * logs.mul(-1).exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if not reverse:
                sldj = sldj + ldj
            else:
                sldj = sldj - ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if not reverse:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        else:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)

        return x, ldj

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(AffineCoupling, self).__init__()
        self.network = CNN(in_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.network(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj

class CNN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(CNN, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = nn.BatchNorm2d(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x


class Invertible1x1ConvLU(nn.Module):
    # https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    def __init__(self, num_channels):
        super(Invertible1x1ConvLU, self).__init__()
        self.num_channels = num_channels
        Q = torch.nn.init.orthogonal_(torch.randn(num_channels, num_channels))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self, x):
        """ assemble W from its pieces (P, L, U, S) """
        print('self.s type: {}\t self.s devodce: {}'.format(type(self.S), self.S.device))
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.num_channels, device=x.device))
        print('l type: {}\t l devodce: {}'.format(type(L), L.device))
        U = torch.triu(self.U, diagonal=1).to(x.device)
        print('u type: {}\t u devodce: {}'.format(type(U), U.device))
        print('self.P type: {}\t self.P devodce: {}'.format(type(self.P), self.P.device))
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            W = self._assemble_W(x)
            z = x @ W
            log_det = torch.sum(torch.log(torch.abs(self.S)))
            sldj = sldj + log_det
        else:
            W = self._assemble_W()
            W_inv = torch.inverse(W)
            z = x @ W_inv
            log_det = -torch.sum(torch.log(torch.abs(self.S)))
            sldj = sldj - log_det
        return z, sldj