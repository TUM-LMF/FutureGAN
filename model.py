# =============================================================================
# FutureGAN Model
# =============================================================================

import torch
import torch.nn as nn
from custom_layers import *


# =============================================================================
# Modules
# -----------------------------------------------------------------------------
# code borrows from: https://github.com/nashory/pggan-pytorch

def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, padding='zero', lrelu=True, batch_norm=False, w_norm=False, pixel_norm=False, only=False):

    if padding=='replication':
        layers.append(nn.ReplicationPad3d(pad))
        pad = 0
    if w_norm:  layers.append(EqualizedConv3d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if lrelu:       layers.append(nn.LeakyReLU(0.2))
        else:           layers.append(nn.ReLU())
        if batch_norm:  layers.append(nn.BatchNorm3d(c_out))
        if pixel_norm:  layers.append(PixelwiseNormLayer())
    return layers


def deconv_t(layers, c_in, c_out, k_size, stride=1, pad=0, padding='zero', lrelu=True, batch_norm=False, w_norm=False, pixel_norm=False, only=False):

    if padding=='replication':
        layers.append(nn.ReplicationPad3d(pad))
        pad = 0
    if w_norm:  layers.append(EqualizedConvTranspose3d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.ConvTranspose3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if lrelu:       layers.append(nn.LeakyReLU(0.2))
        else:           layers.append(nn.ReLU())
        if batch_norm:  layers.append(nn.BatchNorm3d(c_out))
        if pixel_norm:  layers.append(PixelwiseNormLayer())
    return layers


def conv(layers, c_in, c_out, k_size, stride=1, pad=0, padding='zero', lrelu=True, batch_norm=False, w_norm=False, d_gdrop=False, pixel_norm=False, only=False):

    if padding=='replication':
        layers.append(nn.ReplicationPad3d(pad))
        pad = 0
    if d_gdrop:         layers.append(GeneralizedDropOut(mode='prop', strength=0.0))
    if w_norm:          layers.append(EqualizedConv3d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:               layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if lrelu:       layers.append(nn.LeakyReLU(0.2))
        else:           layers.append(nn.ReLU())
        if batch_norm:  layers.append(nn.BatchNorm3d(c_out))
        if pixel_norm:  layers.append(PixelwiseNormLayer())
    return layers


def linear(layers, c_in, c_out, sig=True, w_norm=False):

    layers.append(Flatten())
    if w_norm:  layers.append(EqualizedLinear(c_in, c_out))
    else:       layers.append(nn.Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers


def deepcopy_module(module, target):

    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                          # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())         # copy weights
    return new_module


def get_module_names(model):

    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names



# =============================================================================
# FutureGenerator

class FutureGenerator(nn.Module):

    def __init__(self, config):

        super(FutureGenerator, self).__init__()
        self.config = config
        self.batch_norm = config.batch_norm
        self.g_pixelwise_norm = config.g_pixelwise_norm
        self.w_norm = config.w_norm
        self.padding = config.padding
        self.lrelu = config.lrelu
        self.g_tanh = config.g_tanh
        self.d_gdrop = False
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.nframes_in = config.nframes_in
        self.nframes_pred = config.nframes_pred
        self.nframes = self.nframes_in+self.nframes_pred
        self.layer_name_encode = None
        self.layer_name_decode = None
        self.module_names = []
        self.model = self.get_init_gen()


    def middle_block(self):

        ndim = self.ngf
        layers = []

        # encode [N,C,D(nframes_in),H(4),W(4)] --> [N,C,1,4,4]
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
        layers = conv(layers, ndim, ndim, (self.nframes_in,1,1), 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)

        # decode [N,C,1,4,4] --> [N,C,nframes_pred,H(4),W(4)]
        layers = deconv_t(layers, ndim, ndim, (self.nframes_pred,1,1), 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)

        return  nn.Sequential(*layers), ndim


    def intermediate_block_decode(self, resl):

        halving = False
        layer_name = 'intermediate_decode_{}x{}_{}x{}'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2, resl)), int(pow(2, resl)))
        ndim = self.ngf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ngf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)

        layers = []
        if halving:
            layers = deconv_t(layers, ndim*2, ndim, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
        else:
            layers = deconv_t(layers, ndim, ndim, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
        return  nn.Sequential(*layers), ndim, layer_name


    def intermediate_block_encode(self, resl):

        halving = False
        layer_name = 'intermediate_encode_{}x{}_{}x{}'.format(int(pow(2,resl)), int(pow(2,resl)), int(pow(2, resl-1)), int(pow(2, resl-1)))
        ndim = self.ngf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ngf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)

        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
            layers = conv(layers, ndim, ndim*2, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
            layers = conv(layers, ndim, ndim, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)

        return  nn.Sequential(*layers), ndim, layer_name


    def to_rgb_block(self, ndim):

        layers = []
        layers = deconv(layers, ndim, self.nc, 1, 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm, only=True)
        if self.g_tanh:  layers.append(nn.Tanh())
        return nn.Sequential(*layers)


    def from_rgb_block(self, ndim):

        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
        return  nn.Sequential(*layers)


    def get_init_gen(self):

        model = nn.Sequential()
        middle_block, ndim = self.middle_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('middle_block', middle_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model


    def grow_network(self, resl):

        inter_block_encode, ndim_encode, self.layer_name_encode = self.intermediate_block_encode(resl)
        inter_block_decode, ndim_decode, self.layer_name_decode = self.intermediate_block_decode(resl)

        layers_decode = []
        layers_encode = []
        if resl >= 6:
            layers_decode = deconv_t(layers_decode, ndim_decode*2, ndim_decode*2, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
            layers_encode = conv(layers_encode, ndim_encode*2, ndim_encode*2, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)
        else:
            layers_decode = deconv_t(layers_decode, ndim_decode, ndim_decode, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.g_pixelwise_norm)
            layers_encode = conv(layers_encode, ndim_encode, ndim_encode, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, self.g_pixelwise_norm)

        if resl >= 3 and resl <= 9:
            print(' ... growing generator from {}x{} to {}x{} ... '.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block_encode = nn.Sequential()
            prev_block_encode.add_module('low_resl_from_rgb', low_resl_from_rgb)
            prev_block_encode.add_module('low_resl_downsample', nn.Sequential(*layers_encode))

            next_block_encode = nn.Sequential()
            next_block_encode.add_module('high_resl_from_rgb', self.from_rgb_block(ndim_encode))
            next_block_encode.add_module('high_resl_block_encode', inter_block_encode)

            # we make new network since pytorch does not support remove_module()
            new_model = nn.Sequential()
            new_model.add_module('concat_block_encode', Concat(prev_block_encode, next_block_encode))
            new_model.add_module('fadein_block_encode', FadeInLayer(self.config))

            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
            prev_block_decode = nn.Sequential()
            prev_block_decode.add_module('low_resl_upsample', nn.Sequential(*layers_decode))
            prev_block_decode.add_module('low_resl_to_rgb', low_resl_to_rgb)

            next_block_decode = nn.Sequential()
            next_block_decode.add_module('high_resl_block_decode', inter_block_decode)
            next_block_decode.add_module('high_resl_to_rgb', self.to_rgb_block(ndim_decode))

            for name, module in self.model.named_children():
                if name!='to_rgb_block' and name!='from_rgb_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            new_model.add_module('concat_block_decode', Concat(prev_block_decode, next_block_decode))
            new_model.add_module('fadein_block_decode', FadeInLayer(self.config))

            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)


    def flush_network(self):

        try:
            print(' ... flushing generator ... ')
            # make deep copy and paste.
            high_resl_block_encode = deepcopy_module(self.model.concat_block_encode.layer2, 'high_resl_block_encode')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block_encode.layer2, 'high_resl_from_rgb')

            high_resl_block_decode = deepcopy_module(self.model.concat_block_decode.layer2, 'high_resl_block_decode')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block_decode.layer2, 'high_resl_to_rgb')

            # add the high resolution block.
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name_encode, high_resl_block_encode)

            # add middle.
            for name, module in self.model.named_children():
                if name!='concat_block_encode' and name!='fadein_block_encode' and name!='fadein_block_decode' and name!='concat_block_decode':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            # now, add the high resolution block.
            new_model.add_module(self.layer_name_decode, high_resl_block_decode)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)

            self.model = new_model
            self.module_names = get_module_names(self.model)

        except:
            self.model = self.model


    def freeze_layers(self):

        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print(' ... freezing generator`s pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x):

        if self.config.d_cond==True:
            y = self.model(x)
            x = torch.cat([x, y],2)
        else:
            x = self.model(x)

        return x



# =============================================================================
# Discriminator
# -----------------------------------------------------------------------------
# code borrows from: https://github.com/nashory/pggan-pytorch

class Discriminator(nn.Module):

    def __init__(self, config):

        super(Discriminator, self).__init__()
        self.config = config
        self.nframes_pred = self.config.nframes_pred
        self.batch_norm = config.batch_norm
        self.w_norm = config.w_norm
        if self.config.loss=='lsgan':
            self.d_gdrop = True
        else:
            self.d_gdrop = config.d_gdrop
        self.padding = config.padding
        self.lrelu = config.lrelu
        self.d_sigmoid = config.d_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        if self.config.d_cond==False:
            self.nframes = self.config.nframes_pred
        else:
            self.nframes = config.nframes_in+config.nframes_pred
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()


    def last_block(self):

        # add MinibatchStdConcatLayer later.
        ndim = self.ndf
        layers = []
        layers.append(MinibatchStdConcatLayer())
        layers = conv(layers, ndim+1, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        layers = conv(layers, ndim, ndim, (self.nframes,4,4), 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        layers = linear(layers, ndim, 1, sig=self.d_sigmoid, w_norm=self.w_norm)
        return  nn.Sequential(*layers), ndim


    def intermediate_block(self, resl):

        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2,resl)), int(pow(2,resl)), int(pow(2, resl-1)), int(pow(2, resl-1)))
        ndim = self.ndf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ndf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        # new! bugfix
        ndim = int(ndim)
        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
            layers = conv(layers, ndim, ndim*2, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
            layers = conv(layers, ndim, ndim, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)

        return  nn.Sequential(*layers), ndim, layer_name


    def from_rgb_block(self, ndim):

        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        return  nn.Sequential(*layers)


    def get_init_dis(self):

        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model


    def grow_network(self, resl):

        inter_block, ndim, self.layer_name = self.intermediate_block(resl)

        layers = []
        if resl >= 6:
            layers = conv(layers, ndim*2, ndim*2, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        else:
            layers = conv(layers, ndim, ndim, (1,2,2), (1,2,2), 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)

        if resl >= 3 and resl <= 9:
            print(' ... growing discriminator from {}x{} to {}x{} ... '.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)
            prev_block.add_module('low_resl_downsample', nn.Sequential(*layers))

            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block(ndim))
            next_block.add_module('high_resl_block', inter_block)

            new_model = nn.Sequential()
            new_model.add_module('concat_block', Concat(prev_block, next_block))
            new_model.add_module('fadein_block', FadeInLayer(self.config))

            # we make new network since pytorch does not support remove_module()
            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name=='from_rgb_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)


    def flush_network(self):

        try:
            print(' ... flushing discriminator ... ')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_from_rgb')

            # add the high resolution block.
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)

            # add rest.
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model


    def freeze_layers(self):

        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print(' ... freezing discriminator`s pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x):

        x = self.model(x)
        return x
