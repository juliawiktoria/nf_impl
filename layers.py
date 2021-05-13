import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities

class ActivationNormalisation(nn.Module):
    def __init__(self, num_features, scale=1.):
        super(ActivationNormalisation, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6

    def describe(self):
        print('\t\t\t - > Act Norm with {} num_features.'.format(self.num_features))

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
    def __init__(self, num_features, mid_channels):
        super(AffineCoupling, self).__init__()
        self.num_features = num_features
        self.network = CNN(num_features, mid_channels, 2 * num_features)
        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))

    def describe(self):
        print('\t\t\t - > Aff Coupling with {} num_features'.format(self.num_features))

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

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x, reverse=False):
        b, c, h, w = x.size()
        if not reverse:
            # Squeeze
            x = x.view(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * 2 * 2, h // 2, w // 2)
        else:
            # Unsqueeze
            x = x.view(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(b, c // 4, h * 2, w * 2)

        return x

class Invertible1x1ConvLU(nn.Module):
    # https://github.com/y0ast/Glow-PyTorch/blob/master/modules.py
    def __init__(self, num_features, LU_decomposed=True):
        super(Invertible1x1ConvLU, self).__init__()
        w_shape = [num_features, num_features]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        self.num_features = num_features

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye
        
        self.if_LU = LU_decomposed
        self.w_shape = w_shape
    
    def describe(self):
        if self.if_LU:
            print('\t\t\t - > Inverted 1x1 Conv (LU decomposition) with {} num_features'.format(self.num_features))
        else:
            print('\t\t\t - > Inverted 1x1 Conv with {} num_features'.format(self.num_features))

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.if_LU:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if not reverse:
                weight = self.weight
            else:
                weight = torch.inverse(self.weight)
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if not reverse:
                weight = torch.matmul(self.p, torch.matmul(lower, u))
            else:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, cond_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, x_cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)
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


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels):
        super(NN, self).__init__()
        norm_fn = nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_condconv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_condconv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, x_cond):
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = F.relu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x