import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ActivationNormalisation, AffineCoupling, Invertible1x1Conv, Invertible1x1ConvLU


class GlowModel(nn.Module):
    def __init__(self, num_features, hid_layers, num_levels, num_steps):
        super(GlowModel, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.levels = _Level(num_features=4 * num_features,  # RGB image after squeeze
                           hid_layers=hid_layers,
                           num_levels=num_levels,
                           num_steps=num_steps)

    def forward(self, x, reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x = squeeze(x)
        x, sldj = self.levels(x, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj

    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj


class _Level(nn.Module):
    def __init__(self, num_features, hid_layers, num_levels, num_steps):
        super(_Level, self).__init__()
        
        # initialise K flow steps for the level
        self.steps = nn.ModuleList([_Step(num_features=num_features, hid_layers=hid_layers) for _ in range(num_steps)])

        # keep adding levels recursively until the last one is encountered, then add NONE as a stopper
        if num_levels > 1:
            self.next_lvl = _Level(num_features=2 * num_features,
                              hid_layers=hid_layers,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next_lvl = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)

        if self.next_lvl is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next_lvl(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)

        return x, sldj


class _Step(nn.Module):
    def __init__(self, num_features, hid_layers):
        super(_Step, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.normalisation = ActivationNormalisation(num_features)
        self.convolution = Invertible1x1ConvLU(num_features)
        self.coupling = AffineCoupling(num_features // 2, hid_layers)

    def forward(self, x, sldj=None, reverse=False):

        # forward pass of the step - [ActivationNormalisation, Inverted1x1ConvLU, AffineCoupling]
        if not reverse:
            x, sldj = self.normalisation(x, sldj, reverse)
            x, sldj = self.convolution(x, sldj, reverse)
            x, sldj = self.coupling(x, sldj, reverse)
        
        # reversed pass of the step - [AffineCoupling, Inverted1x1ConvLU, ActivationNormalisation]
        else:
            x, sldj = self.coupling(x, sldj, reverse)
            x, sldj = self.convolution(x, sldj, reverse)
            x, sldj = self.normalisation(x, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x
