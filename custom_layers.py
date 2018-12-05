# =============================================================================
# Custom FutureGAN Layers
# -----------------------------------------------------------------------------             
# code borrows from: 
# https://github.com/nashory/pggan-pytorch
# https://github.com/tkarras/progressive_growing_of_gans
# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal, calculate_gain
from torch.autograd import Variable


class Concat(nn.Module):
    '''
    same function as ConcatTable container in Torch7
    '''
    
    def __init__(self, layer1, layer2):
        super(Concat, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        
        
    def forward(self,x):
        y = [self.layer1(x), self.layer2(x)]
        return y



class Flatten(nn.Module):
    
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, x):
        return x.view(x.size(0), -1)
      
    
    
class FadeInLayer(nn.Module):
    
    def __init__(self, config):
        super(FadeInLayer, self).__init__()
        self.alpha = 0.0


    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))


    # input : [x_low, x_high] from ConcatTable()
    def forward(self, x):
        return torch.add(x[0].mul(1.0-self.alpha), x[1].mul(self.alpha))
    
    

class MinibatchStdConcatLayer(nn.Module):
    
    def __init__(self, averaging='all'):
        super(MinibatchStdConcatLayer, self).__init__()
        self.averaging = averaging
        
        
    def forward(self, x):
        s = x.size()                                    # [NCDHW] Input shape.
        y = x
        y = y-torch.mean(y, 0, keepdim=True)            # [NCDHW] Subtract mean over group.
        y = torch.mean(torch.pow(y,2), 0, keepdim=True) # [NCDHW] Calc variance over group.
        y = torch.sqrt(y + 1e-8)                        # [NCDHW] Calc stddev over group.
        for axis in [1,2,3,4]:
            y = torch.mean(y, int(axis), keepdim=True)  # [N1111] Take average over fmaps and pixels.
        y = y.expand(s[0], 1, s[2], s[3], s[4])         # [N1DHW] Replicate over group and pixels.
        x = torch.cat([x, y], 1)                        # [NCHW] Append as new fmap.
        return x
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)



class PixelwiseNormLayer(nn.Module):
    
    def __init__(self):
        super(PixelwiseNormLayer, self).__init__()
        self.eps = 1e-8


    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5
  
   
    
class EqualizedConv3d(nn.Module):
    
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
        super(EqualizedConv3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.conv.weight, a=calculate_gain('conv3d'))
        elif initializer == 'xavier':   xavier_normal(self.conv.weight)
            
        self.conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)


    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1,1).expand_as(x)
 
    
      
class EqualizedConvTranspose3d(nn.Module):
  
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(EqualizedConvTranspose3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.deconv.weight, a=calculate_gain('conv3d'))
        elif initializer == 'xavier':   xavier_normal(self.deconv.weight)
        
        self.deconv_w = self.deconv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data/self.scale)
        
        
    def forward(self, x):
        x = self.deconv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1,1).expand_as(x)
        
    
    
class EqualizedLinear(nn.Module):
    
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(EqualizedLinear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.linear.weight)
        
        self.linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)
        
        
    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1,-1).expand_as(x)



class GeneralizedDropOut(nn.Module):
    '''
    This is only important for really easy datasets or LSGAN, 
    adding noise to discriminator to prevent discriminator 
    from spiraling out of control for too easy datasets. 
    '''
    
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(GeneralizedDropOut, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None


    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd


    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str