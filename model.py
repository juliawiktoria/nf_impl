import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

# add a posibility of creating a transform list instead of hard-coded ones
# class for building GlowModel, not to be used on its own
class _Step(nn.Module):
    def __init__(self, num_features, hid_layers, step_number):
        super(_Step, self).__init__()

        self.step_id = step_number
        # add transforms to a step
        self.normalisation = ActivationNormalisation(num_features)
        # every other step the convolution is LU decomposed
        self.convolution = Invertible1x1ConvLU(num_features, LU_decomposed=(step_number % 2 ==0))
        self.coupling = AffineCoupling(num_features // 2, hid_layers)

    def describe(self):
        print('\t\t - > STEP {}'.format(self.step_id))
        self.normalisation.describe()
        self.convolution.describe()
        self.coupling.describe()

    def forward(self, x, sum_lower_det_jacobian=None, reverse=False):

        # forward pass of the step - [ActivationNormalisation, Inverted1x1ConvLU, AffineCoupling]
        if not reverse:
            x, sum_lower_det_jacobian = self.normalisation(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.convolution(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.coupling(x, sum_lower_det_jacobian, reverse)
        
        # reversed pass of the step - [AffineCoupling, Inverted1x1ConvLU, ActivationNormalisation]
        else:
            x, sum_lower_det_jacobian = self.coupling(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.convolution(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.normalisation(x, sum_lower_det_jacobian, reverse)

        return x, sum_lower_det_jacobian


# class for building GlowModel, not to be used on its own
class _Level(nn.Module):
    # creates a chain of levels
    # level comprises of a squeeze step, K flow steps, and split step (except for the last leves, which does not have a split step)
    def __init__(self, num_features, hid_layers, num_levels, num_steps, lvl_number):
        super(_Level, self).__init__()
        
         # create K steps of the flow K x ([t,t,t]) where t is a flow transform
        # channels (features) are multiplied by 4 to account for squeeze operation that takes place before flow steps
        self.flow_steps = nn.ModuleList([_Step(num_features=num_features, hid_layers=hid_layers, step_num=i+1) for i in range(num_steps)])

        # level id number
        self.lvl_id = lvl_number

        # keep adding levels recursively until the last one is encountered, then add NONE as a stopper
        if num_levels > 1:
            self.next_lvl = _Level(num_features=2 * num_features, hid_layers=hid_layers, num_levels=num_levels - 1, num_steps=num_steps, lvl_number=self.lvl_id + 1)
        else:
            self.next_lvl = None

        # squeeze object
        self.squeeze = Squeeze()

    def describe(self):
        print('\t - > Level {}'.format(self.lvl_id))
        print('\t\t - > Squeeze layer')
        for step in self.flow_steps:
            step.describe()
        if self.next_lvl is not None:
            print('\t\t - > Split layer')
            self.next_lvl.describe()

    def forward(self, x, sum_lower_det_jacobian, reverse=False):
        if not reverse:
            for step in self.flow_steps:
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)

        # recursively calling next levels until there are no more
        if self.next_lvl is not None:
            x = self.squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sum_lower_det_jacobian = self.next_lvl(x, sum_lower_det_jacobian, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = self.squeeze(x, reverse=True)

        if reverse:
            # reversing the steps
            for step in reversed(self.flow_steps):
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)

        return x, sum_lower_det_jacobian

# the whole model
class GlowModel(nn.Module):
    def __init__(self, num_features, hid_layers, num_levels, num_steps):
        super(GlowModel, self).__init__()

        # Use bounds to rescale images before converting to logits, not learned
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.levels = _Level(num_features=4 * num_features, hid_layers=hid_layers, num_levels=num_levels, num_steps=num_steps, lvl_number=1)

        self.squeeze = Squeeze()
    
    def describe(self):
        # method for describing the rchitecture of the model, when called it calls all its subparts
        # and produces a nice visualisation of the levels, steps, and transforms
        print('==============GLOW MODEL============')
        self.levels.describe()
        print('====================================')

    def forward(self, x, reverse=False):
        if not reverse:
            # the model takes input between 0 and 1
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max {}/{}'.format(x.min(), x.max()))
            # De-quantize and convert to logits
            x, sum_lower_det_jacobian = self._pre_process(x)
        else:
            sum_lower_det_jacobian = torch.zeros(x.size(0), device=x.device)

        x = self.squeeze(x)
        # reverse operation are solved in the levels and steps
        x, sum_lower_det_jacobian = self.levels(x, sum_lower_det_jacobian, reverse)
        x = self.squeeze(x, reverse=True)

        return x, sum_lower_det_jacobian

    # pre-processing input
    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sum_lower_det_jacobian = ldj.flatten(1).sum(-1)

        return y, sum_lower_det_jacobian