# =============================================================================
# Train FutureGAN
# =============================================================================

'''
Script to train FutureGAN.

Your training data is assumed to be arranged in this way:
    data_root/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.

To resume training from a checkpoint, set the --use_ckpt=`True` and specify --ckpt_path:
    --ckpt_path=`path_to_generator_ckpt` [0]
    --ckpt_path=`path_to_discriminator_ckpt` [1]

For further options and information, read the provided `help` information of the optional arguments below.

-------------------------------------------------------------------------------
This code borrows from:
    https://github.com/nashory/pggan-pytorch
    https://github.com/tkarras/progressive_growing_of_gans
    https://github.com/github-pengge/PyTorch-progressive_growing_of_gans

The implementation of the wgan-gp loss borrows from:
    https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py and
    https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
'''

import os
import time
import argparse
from PIL import Image
from math import floor, ceil
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torchvision.transforms as transforms
from utils import save_video_grid, count_model_params
from video_dataset import VideoFolder, video_loader
from torch.utils.data import DataLoader
import model as model



# =============================================================================
# config options

help_description = 'This script trains a FutureGAN model for video prediction according to the specified optional arguments.'

parser = argparse.ArgumentParser(description=help_description)

# general
parser.add_argument('--dgx', type=bool, default=False, help='set to True, if code is run on dgx, default=`False`')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus for (multi-)gpu training, default=1')
parser.add_argument('--random_seed', type=int, default=int(time.time()), help='seed for generating random numbers, default = `int(time.time())`')
parser.add_argument('--ext', action='append', default=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'], help='list of strings of allowed file extensions, default=[`.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`, `.pgm`]')
parser.add_argument('--use_ckpt', type=bool, default=False, help='continue training from checkpoint, default=`False`')

parser.add_argument('--ckpt_path', action='append', help='list of path(s) to training checkpoints to continue training or for testing, [0] Generator and [1] Discriminator, default=``')
parser.add_argument('--data_root', type=str, default='', help='path to root directory of training data (ex. -->path_to_dataset/train)')
parser.add_argument('--log_dir', type=str, default='./logs', help='path to directory of log files')
parser.add_argument('--experiment_name', type=str, default='', help='name of experiment (if empty, current date and time will be used), default=``')

parser.add_argument('--d_cond', type=bool, default=True, help='condition discriminator on input frames, default=`True`')
parser.add_argument('--nc', type=int, default=3, help='number of input image color channels, default=3')
parser.add_argument('--max_resl', type=int, default=128, help='max. frame resolution --> image size: max_resl x max_resl , default=128')
parser.add_argument('--nframes_in', type=int, default=6, help='number of input video frames in one sample, default=12')
parser.add_argument('--nframes_pred', type=int, default=6, help='number of video frames to predict in one sample, default=6')
# p100
parser.add_argument('--batch_size_table', type=dict, default={4:32, 8:16, 16:8, 32:4, 64:2, 128:1, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
## dgx
#parser.add_argument('--batch_size_table', type=dict, default={4:256, 8:128, 16:64, 32:32, 64:16, 128:8, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
parser.add_argument('--trns_tick', type=int, default=10, help='number of epochs for transition phase, default=10')
parser.add_argument('--stab_tick', type=int, default=10, help='number of epochs for stabilization phase, default=10')

# training
parser.add_argument('--nz', type=int, default=512, help='dimension of input noise vector z, default=512')
parser.add_argument('--ngf', type=int, default=512, help='feature dimension of final layer of generator, default=512')
parser.add_argument('--ndf', type=int, default=512, help='feature dimension of first layer of discriminator, default=512')

parser.add_argument('--loss', type=str, default='wgan_gp', help='which loss functions to use (choices: `gan`, `lsgan` or `wgan_gp`), default=`wgan_gp`')
parser.add_argument('--d_eps_penalty', type=bool, default=True, help='adding an epsilon penalty term to wgan_gp loss to prevent loss drift (eps=0.001), default=True')
parser.add_argument('--acgan', type=bool, default=False, help='adding a label penalty term to wgan_gp loss --> makes GAN conditioned on classification labels of dataset, default=False')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type, default=adam')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lr_decay', type=float, default=0.87, help='learning rate decay at every resolution transition, default=0.87')

parser.add_argument('--lrelu', type=bool, default=True, help='use leaky relu instead of relu, default=True')
parser.add_argument('--padding', type=str, default='zero', help='which padding to use (choices: `zero`, `replication`), default=`zero`')
parser.add_argument('--w_norm', type=bool, default=True, help='use weight scaling, default=True')
parser.add_argument('--batch_norm', type=bool, default=False, help='use batch-normalization (not recommended), default=False')
parser.add_argument('--g_pixelwise_norm', type=bool, default=True, help='use pixelwise normalization for generator, default=True')
parser.add_argument('--d_gdrop', type=bool, default=False, help='use generalized dropout layer (inject multiplicative Gaussian noise) for discriminator when using LSGAN loss, default=False')
parser.add_argument('--g_tanh', type=bool, default=False, help='use tanh at the end of generator, default=False')
parser.add_argument('--d_sigmoid', type=bool, default=False, help='use sigmoid at the end of discriminator, default=False')
parser.add_argument('--x_add_noise', type=bool, default=False, help='add noise to the real image(x) when using LSGAN loss, default=False')
parser.add_argument('--z_pixelwise_norm', type=bool, default=False, help='if mode=`gen`: pixelwise normalization of latent vector (z), default=False')

# display and save
parser.add_argument('--tb_logging', type=bool, default=False, help='enable tensorboard visualization, default=True')
parser.add_argument('--update_tb_every', type=int, default=100, help='display progress every specified iteration, default=100')
parser.add_argument('--save_img_every', type=int, default=100, help='save images every specified iteration, default=100')
parser.add_argument('--save_ckpt_every', type=int, default=5, help='save checkpoints every specified epoch, default=5')

# parse and save training config
config = parser.parse_args()

# current time is used to name folders and files if --experiment_name is not specified
current_time = time.strftime('%Y-%m-%d_%H%M%S')


# =============================================================================
#  import Logger if --tb_logging==True
if  config.tb_logging:
    from tb_logger import Logger


# =============================================================================
# enable cuda if gpu(s) is/are available

if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')



# =============================================================================
# training routine

class Trainer:
    '''
    Class to train a FutureGAN model.

    Data is assumed to be arranged in this way:
        data_root/video/frame.ext -> dataset/train/video1/frame1.ext
                                                  -> dataset/train/video1/frame2.ext
                                                  -> dataset/train/video2/frame1.ext
                                                  -> ...
    '''

    def __init__(self, config):

        self.config = config

        # log directory
        if self.config.experiment_name=='':
            self.experiment_name = current_time
        else:
            self.experiment_name = self.config.experiment_name

        self.log_dir = config.log_dir+'/'+self.experiment_name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # save config settings to file
        with open(self.log_dir+'/train_config.txt','w') as f:
            print('------------- training configuration -------------', file=f)
            for k, v in vars(config).items():
                print(('{}: {}').format(k, v), file=f)
            print(' ... loading training configuration ... ')
            print(' ... saving training configuration to {}'.format(f))

        self.train_data_root = self.config.data_root

        # training samples
        self.train_sample_dir = self.log_dir+'/samples_train'

        # checkpoints
        self.ckpt_dir = self.log_dir+'/ckpts'

        # tensorboard
        if self.config.tb_logging:
            self.tb_dir = self.log_dir+'/tensorboard'

        self.use_cuda = use_cuda
        self.nz = config.nz
        self.nc = config.nc
        self.optimizer = config.optimizer
        self.batch_size_table = config.batch_size_table
        self.lr = config.lr
        self.d_eps_penalty = config.d_eps_penalty
        self.acgan = config.acgan
        self.max_resl = int(np.log2(config.max_resl))
        self.nframes_in = config.nframes_in
        self.nframes_pred = config.nframes_pred
        self.nframes = self.nframes_in+self.nframes_pred
        self.ext = config.ext
        self.nworkers = 4
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.complete = 0.0
        self.x_add_noise = config.x_add_noise
        self.fadein = {'G':None, 'D':None}
        self.init_resl = 2
        self.init_img_size = int(pow(2, self.init_resl))

        # initialize model G as FutureGenerator from model.py
        self.G = model.FutureGenerator(config)

        # initialize model D as Discriminator from model.py
        self.D = model.Discriminator(config)

        # define losses
        if self.config.loss=='lsgan':
            self.criterion = torch.nn.MSELoss()

        elif self.config.loss=='gan':
            if self.config.d_sigmoid==True:
                self.criterion = torch.nn.BCELoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()

        elif self.config.loss=='wgan_gp':
            if self.config.d_sigmoid==True:
                self.criterion = torch.nn.BCELoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise Exception('Loss is undefined! Please set one of the following: `gan`, `lsgan` or `wgan_gp`')

        # check --use_ckpt
        # if --use_ckpt==False: build initial model
        # if --use_ckpt==True: load and build model from specified checkpoints
        if self.config.use_ckpt==False:

            print(' ... creating initial models ... ')

            # set initial model parameters
            self.resl = self.init_resl
            self.start_resl = self.init_resl
            self.globalIter = 0
            self.nsamples = 0
            self.stack = 0
            self.epoch = 0
            self.iter_start = 0
            self.phase = 'init'
            self.flag_flush = False

            # define tensors, ship model to cuda, and get dataloader
            self.renew_everything()

            # count model parameters
            nparams_g = count_model_params(self.G)
            nparams_d = count_model_params(self.D)

            # save initial model structure to file
            with open(self.log_dir+'/initial_model_structure_{}x{}.txt'.format(self.init_img_size, self.init_img_size),'w') as f:
                print('--------------------------------------------------', file=f)
                print('Sequences in Dataset: ', len(self.dataset), ', Batch size: ', self.batch_size, file=f)
                print('Global iteration step: ', self.globalIter, ', Epoch: ', self.epoch, file=f)
                print('Phase: ', self.phase, file=f)
                print('Number of Generator`s model parameters: ', file=f)
                print(nparams_g, file=f)
                print('Number of Discriminator`s model parameters: ', file=f)
                print(nparams_d, file=f)
                print('--------------------------------------------------', file=f)
                print('Generator structure: ', file=f)
                print(self.G, file=f)
                print('--------------------------------------------------', file=f)
                print('Discriminator structure: ', file=f)
                print(self.D, file=f)
                print('--------------------------------------------------', file=f)
                print(' ... initial models have been built successfully ... ')
                print(' ... saving initial model strutures to {}'.format(f))

            # ship everything to cuda and parallelize for ngpu>1
            if self.use_cuda:
                self.criterion = self.criterion.cuda()
                torch.cuda.manual_seed(config.random_seed)
                if config.ngpu==1:
                    self.G = torch.nn.DataParallel(self.G).cuda(device=0)
                    self.D = torch.nn.DataParallel(self.D).cuda(device=0)
                else:
                    gpus = []
                    for i  in range(config.ngpu):
                        gpus.append(i)
                    self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda()
                    self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()

        else:

            # re-ship everything to cuda
            if self.use_cuda:
                self.G = self.G.cuda()
                self.D = self.D.cuda()

            # load checkpoint
            print(' ... loading models from checkpoints ... {} and {}'.format(self.config.ckpt_path[0], self.config.ckpt_path[1]))
            self.ckpt_g = torch.load(self.config.ckpt_path[0])
            self.ckpt_d = torch.load(self.config.ckpt_path[1])

            # get model parameters
            self.resl = self.ckpt_g['resl']
            self.start_resl = int(self.ckpt_g['resl'])
            self.iter_start = self.ckpt_g['iter']+1
            self.globalIter = int(self.ckpt_g['globalIter'])
            self.stack = int(self.ckpt_g['stack'])
            self.nsamples = int(self.ckpt_g['nsamples'])
            self.epoch = int(self.ckpt_g['epoch'])
            self.fadein['G'] = self.ckpt_g['fadein']
            self.fadein['D'] = self.ckpt_d['fadein']
            self.phase = self.ckpt_d['phase']
            self.complete  = self.ckpt_d['complete']
            self.flag_flush = self.ckpt_d['flag_flush']
            img_size = int(pow(2, floor(self.resl)))

            # get model structure
            self.G = self.ckpt_g['G_structure']
            self.D = self.ckpt_d['D_structure']

            # define tensors, ship model to cuda, and get dataloader
            self.renew_everything()
            self.schedule_resl()
            self.nsamples = int(self.ckpt_g['nsamples'])

            # save loaded model structure to file
            with open(self.log_dir+'/resumed_model_structure_{}x{}.txt'.format(img_size, img_size),'w') as f:
                print('--------------------------------------------------', file=f)
                print('Sequences in Dataset: ', len(self.dataset), file=f)
                print('Global iteration step: ', self.globalIter, ', Epoch: ', self.epoch, file=f)
                print('Phase: ', self.phase, file=f)
                print('--------------------------------------------------', file=f)
                print('Reloaded Generator structure: ', file=f)
                print(self.G, file=f)
                print('--------------------------------------------------', file=f)
                print('Reloaded Discriminator structure: ', file=f)
                print(self.D, file=f)
                print('--------------------------------------------------', file=f)
                print(' ... models have been loaded successfully from checkpoints ... ')
                print(' ... saving resumed model strutures to {}'.format(f))

            # load model state_dict
            self.G.load_state_dict(self.ckpt_g['state_dict'])
            self.D.load_state_dict(self.ckpt_d['state_dict'])

            # load optimizer state dict
            lr = self.lr
            for i in range(1,int(floor(self.resl))-1):
                self.lr = lr*(self.config.lr_decay**i)
#            self.opt_g.load_state_dict(self.ckpt_g['optimizer'])
#            self.opt_d.load_state_dict(self.ckpt_d['optimizer'])
#            for param_group in self.opt_g.param_groups:
#                self.lr = param_group['lr']

        # tensorboard logging
        self.tb_logging = self.config.tb_logging
        if self.tb_logging==True:
            if not os.path.exists(self.tb_dir):
                os.makedirs(self.tb_dir)
            self.logger = Logger(self.tb_dir)


    def schedule_resl(self):

        # trns and stab if resl > 2
        if floor(self.resl)!=2:
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick

        # alpha and delta parameters for smooth fade-in (resl-interpolation)
        delta = 1.0/(self.trns_tick+self.stab_tick)
        d_alpha = 1.0*self.batch_size/self.trns_tick/len(self.dataset)

        # update alpha if FadeInLayer exist
        if self.fadein['D'] is not None:
            if self.resl%1.0 < (self.trns_tick)*delta:
                self.fadein['G'][0].update_alpha(d_alpha)
                self.fadein['G'][1].update_alpha(d_alpha)
                self.fadein['D'].update_alpha(d_alpha)
                self.complete = self.fadein['D'].alpha*100
                self.phase = 'trns'
            elif self.resl%1.0 >= (self.trns_tick)*delta and self.phase != 'final':
                self.phase = 'stab'

        # increase resl linearly every tick
        prev_nsamples = self.nsamples
        self.nsamples = self.nsamples + self.batch_size
        if (self.nsamples%len(self.dataset)) < (prev_nsamples%len(self.dataset)):
            self.nsamples = 0

            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network.
            if self.flag_flush and self.resl%1.0 >= (self.trns_tick)*delta and prev_resl!=2:
                if self.fadein['D'] is not None:
                    self.fadein['G'][0].update_alpha(d_alpha)
                    self.fadein['G'][1].update_alpha(d_alpha)
                    self.fadein['D'].update_alpha(d_alpha)
                    self.complete = self.fadein['D'].alpha*100
                self.flag_flush = False
                self.G.module.flush_network()   # flush G
                self.D.module.flush_network()   # flush and,
                self.fadein['G'] = None
                self.fadein['D'] = None
                self.complete = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'stab'
                self.print_model_structure()

            # grow network.
            if floor(self.resl) != prev_resl and floor(self.resl)<self.max_resl+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.module.grow_network(floor(self.resl))
                self.D.module.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['G'] = [self.G.module.model.fadein_block_decode, self.G.module.model.fadein_block_encode]
                self.fadein['D'] = self.D.module.model.fadein_block
                self.flag_flush = True
                self.print_model_structure()

            if floor(self.resl) >= self.max_resl and self.resl%1.0 >= self.trns_tick*delta:
                self.phase = 'final'
                self.resl = self.max_resl+self.trns_tick*delta


    def print_model_structure(self):

        img_size = self.img_size

        # count model parameters
        nparams_g = count_model_params(self.G)
        nparams_d = count_model_params(self.D)

        with open(self.log_dir+'/model_structure_{}x{}.txt'.format(img_size, img_size),'a') as f:
            print('--------------------------------------------------', file=f)
            print('Sequences in Dataset: ', len(self.dataset), file=f)
            print('Global iteration step: ', self.globalIter, ', Epoch: ', self.epoch, file=f)
            print('Phase: ', self.phase, file=f)
            print('Number of Generator`s model parameters: ', file=f)
            print(nparams_g, file=f)
            print('Number of Discriminator`s model parameters: ', file=f)
            print(nparams_d, file=f)
            print('--------------------------------------------------', file=f)
            print('New Generator structure: ', file=f)
            print(self.G.module, file=f)
            print('--------------------------------------------------', file=f)
            print('New Discriminator structure: ', file=f)
            print(self.D.module, file=f)
            print('--------------------------------------------------', file=f)
            print(' ... models are being updated ... ')
            print(' ... saving updated model strutures to {}'.format(f))


    def renew_everything(self):

        # renew dataloader
        self.img_size = int(pow(2,min(floor(self.resl), self.max_resl)))
        self.batch_size = int(self.batch_size_table[pow(2,min(floor(self.resl), self.max_resl))])
        self.video_loader = video_loader
        self.transform_video = transforms.Compose([transforms.Resize(size=(self.img_size,self.img_size), interpolation=Image.NEAREST), transforms.ToTensor(),])
        self.dataset = VideoFolder(video_root=self.train_data_root, video_ext=self.ext, nframes=self.nframes, loader=self.video_loader, transform=self.transform_video)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.nworkers)
        self.epoch_tick = int(ceil(len(self.dataset)/self.batch_size))

        # define tensors
        self.real_label = Variable(torch.FloatTensor(self.batch_size, 1).fill_(1))
        self.fake_label = Variable(torch.FloatTensor(self.batch_size, 1).fill_(0))

        # wrapping autograd Variable.
        self.z = Variable(torch.FloatTensor(self.batch_size, self.nc, self.nframes_in, self.img_size, self.img_size))
        self.x = Variable(torch.FloatTensor(self.batch_size, self.nc, self.nframes, self.img_size, self.img_size))
        self.x_gen = Variable(torch.FloatTensor(self.batch_size, self.nc, self.nframes_pred, self.img_size, self.img_size))
        self.z_x_gen = Variable(torch.FloatTensor(self.batch_size, self.nc, self.nframes, self.img_size, self.img_size))

        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.x = self.x.cuda()
            self.x_gen = self.x_gen.cuda()
            self.z_x_gen = self.z_x_gen.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)


    def feed_interpolated_input(self, x):

        # interpolate input to match network resolution
        if self.phase == 'Gtrns' and floor(self.resl)>2 and floor(self.resl)<=self.max_resl:
            alpha = self.complete/100.0
            transform = transforms.Compose( [   transforms.ToPILImage(),
                                                transforms.Resize(size=int(pow(2,floor(self.resl)-1)), interpolation=0),      # 0: nearest
                                                transforms.Resize(size=int(pow(2,floor(self.resl))), interpolation=0),      # 0: nearest
                                                transforms.ToTensor(),
                                            ] )

            x_low = x.clone().add(1).mul(0.5)
            for i in range(x_low.size(0)):
                for j in range(x_low.size(2)):
                    x_low[i,:,j,:,:] = transform(x_low[i,:,j,:,:]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1-alpha))

        if self.use_cuda:
            return x.cuda()
        else:
            return x


    def add_noise(self, x):

        if self.x_add_noise==False:
            return x

        # add noise to variable
        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.x_gen_label).data[0] * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z


    def get_batch(self):

        dataIter = iter(self.dataloader)
        return next(dataIter)


    def train(self):

        # train loop
        for step in range(self.start_resl, self.max_resl+2):

            for iter in tqdm(range(self.iter_start,(self.trns_tick+self.stab_tick)*int(ceil(len(self.dataset)/self.batch_size)))):

                self.iter = iter
                self.globalIter = self.globalIter+1
                self.stack = self.stack + self.batch_size
                if self.stack > ceil(len(self.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = 0

                   # save ckpt
                    if self.epoch%self.config.save_ckpt_every==0:
                        self.save_ckpt(self.ckpt_dir)

                # schedule resolution and update parameters
                self.schedule_resl()

                # zero gradients
                self.G.zero_grad()
                self.D.zero_grad()

                # interpolate discriminator real input
                self.x.data = self.feed_interpolated_input(self.get_batch())

                # if 'x_add_noise' --> input to generator without noise, input to discriminator with noise
                self.z.data = self.x.data[:,:,:self.nframes_in,:,:]
                if self.x_add_noise:
                    self.x = self.add_noise(self.x)
                if self.config.d_cond:
                    self.z_x_gen = self.G(self.z)
                    self.x_gen.data = self.z_x_gen.data[:,:,self.nframes_in:,:,:]
                    self.x_label = self.D(self.x.detach())
                    self.x_gen_label = self.D(self.z_x_gen.detach())
                else:
                    self.x_gen = self.G(self.z)
                    self.z_x_gen.data[:,:,:self.nframes_in,:,:] = self.z.data
                    self.z_x_gen.data[:,:,self.nframes_in:,:,:] = self.x_gen.data
                    self.x_label = self.D(self.x[:,:,self.nframes_in:,:,:].detach())
                    self.x_gen_label = self.D(self.x_gen.detach())

                # mse loss
                if self.config.loss=='lsgan':
                    loss_d = self.criterion(self.x_label, self.real_label) + self.criterion(self.x_gen_label, self.fake_label)

                # cross entropy with logits loss
                elif self.config.loss=='gan':
                    loss_d = self.criterion(self.x_label, self.real_label) + self.criterion(self.x_gen_label, self.fake_label)

                # wgan-gp loss
                elif self.config.loss=='wgan_gp':
                    loss_d = torch.mean(self.x_gen_label)-torch.mean(self.x_label)

                    # gradient penalty
                    lam = 10
                    alpha = torch.rand(self.batch_size, 1)
                    if self.config.d_cond==False:
                        alpha = alpha.expand(self.batch_size, self.x[:,:,self.nframes_in:,:,:][0].nelement()).contiguous().view(self.batch_size, self.x.size(1), self.x[:,:,self.nframes_in:,:,:].size(2), self.x.size(3), self.x.size(4))
                    else:
                        alpha = alpha.expand(self.batch_size, self.x[0].nelement()).contiguous().view(self.batch_size, self.x.size(1), self.x.size(2), self.x.size(3), self.x.size(4))
                    if self.use_cuda:
                        alpha = alpha.cuda()
                    if self.config.d_cond:
                        interpolates = alpha*self.x.data+((1-alpha)*self.z_x_gen.data)
                    else:
                        interpolates = alpha*self.x[:,:,self.nframes_in:,:,:].data+((1-alpha)*self.x_gen.data)
                    if self.use_cuda:
                        interpolates = interpolates.cuda()
                    interpolates = Variable(interpolates, requires_grad=True)
                    interpolates_label = self.D(interpolates)
                    gradients = torch.autograd.grad(outputs=interpolates_label, inputs=interpolates,
                              grad_outputs=torch.ones(interpolates_label.size()).cuda() if self.use_cuda else torch.ones(interpolates_label.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
#                    gradients = torch.autograd.grad(outputs=interpolates_label.sum().cuda() if self.use_cuda else interpolates_label.sum(), inputs=interpolates, create_graph=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
                    loss_d = loss_d+lam*gradient_penalty

                    # epsilon penalty
                    if self.d_eps_penalty==True:
                        eps = 0.001
                        eps_penalty = torch.mean((self.x_label-0)**2)
                        loss_d = loss_d+eps_penalty*eps

                    # label penalty
                    # !!! makes GAN conditioned on classification labels of dataset
                    # only makes sense, if actual labels are given, which is not the case here.
                    if self.acgan==True:
                        cond_weight_d = 1.0
                        label_penalty_d = self.criterion(self.x_gen_label, self.fake_label)+self.criterion(self.x_label, self.real_label)
                        loss_d = loss_d+label_penalty_d*cond_weight_d

                # update discriminator
                loss_d.backward()
                self.opt_d.step()

                # get discriminator output
                if self.config.d_cond:
                    self.x_gen_label = self.D(self.z_x_gen)
                else:
                    self.x_gen_label = self.D(self.x_gen)

                # mse loss
                if self.config.loss=='lsgan':
                    loss_g = self.criterion(self.x_gen_label, self.real_label.detach())

                # cross entropy with logits loss
                elif self.config.loss=='gan':
                    loss_g = self.criterion(self.x_gen_label, self.real_label.detach())

                # wgan loss
                elif self.config.loss=='wgan_gp':
                    loss_g = -torch.mean(self.x_gen_label)

                    # label penalty
                    if self.acgan==True:
                        cond_weight_g = 1.0
                        label_penalty_g = self.criterion(self.x_gen_label, self.fake_label)
                        loss_g = loss_g+label_penalty_g*cond_weight_g

                # update generator
                loss_g.backward()
                self.opt_g.step()

                # set max. nr of samples for saving video grid logs
                if self.batch_size >= 8:
                    k = 8
                else:
                    k = self.batch_size

                # save video grid logs
                if self.globalIter%self.config.save_img_every==0 or self.globalIter==1:

                    # log x, z_x_gen
                    if not os.path.exists(self.train_sample_dir):
                        os.makedirs(self.train_sample_dir)

                    # save video grid: x, z_x_gen images
                    save_video_grid(self.x.data[:k,:,:,:,:], self.train_sample_dir+'/'+'x_E{}_I{}_R{}x{}_{}_G{}_D{}.jpg'.format(int(self.epoch), int(self.globalIter), int(self.img_size), int(self.img_size), self.phase, self.complete, self.complete))
                    save_video_grid(self.z_x_gen.data[:k,:,:,:,:], self.train_sample_dir+'/'+'z_x_gen_E{}_I{}_R{}x{}_{}_G{}_D{}.jpg'.format(int(self.epoch), int(self.globalIter), int(self.img_size), int(self.img_size), self.phase, self.complete, self.complete))

                # save tensorboard logs
                if self.tb_logging==True:

                    if self.globalIter%self.config.update_tb_every==0 or self.globalIter==1:

                        # log loss_g and loss_d
                        self.logger.log_scalar('loss/G', loss_g.data[0], self.globalIter)
                        self.logger.log_scalar('loss/D', loss_d.data[0], self.globalIter)

                        # log resl, lr and epoch
                        self.logger.log_scalar('tick/resl', int(pow(2,floor(self.resl))), self.globalIter)
                        self.logger.log_scalar('tick/lr', self.lr, self.globalIter)
                        self.logger.log_scalar('tick/epoch', self.epoch, self.globalIter)

                    # log model parameter histograms weight, bias, weight.grad and bias.grad
                    if self.globalIter%(self.config.update_tb_every*10)==0 or self.globalIter==1:
                        for tag, value in self.G.named_parameters():
                            tag = tag.replace('.', '/')
                            self.logger.log_histogram('G/'+tag, self.var2np(value), self.globalIter)
                            if value.grad is not None:
                                self.logger.log_histogram('G/'+tag+'/grad', self.var2np(value.grad), self.globalIter)
                        for tag, value in self.D.named_parameters():
                            tag = tag.replace('.', '/')
                            self.logger.log_histogram('D/'+tag, self.var2np(value), self.globalIter)
                            if value.grad is not None:
                                self.logger.log_histogram('D/'+tag+'/grad', self.var2np(value.grad), self.globalIter)

            self.iter_start=0

        # save final model
        self.save_final_model(self.log_dir)


    def get_state(self, target):

        # ship models to cpu
        self.G_save = self.G.cpu()
        self.D_save = self.D.cpu()

        if target == 'G':
            state = {
                'G_structure': self.G_save,
                'globalIter': self.globalIter,
                'nsamples': self.nsamples,
                'stack': self.stack,
                'epoch': self.epoch,
                'resl' : self.resl,
                'iter': self.iter,
                'state_dict' : self.G_save.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
                'fadein'    : self.fadein['G'],
                'phase'     : self.phase,
                'complete': self.complete,
                'flag_flush': self.flag_flush,
            }
            return state

        elif target == 'D':
            state = {
                'D_structure': self.D_save,
                'globalIter': self.globalIter,
                'nsamples': self.nsamples,
                'stack': self.stack,
                'epoch': self.epoch,
                'resl' : self.resl,
                'iter': self.iter,
                'state_dict' : self.D_save.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
                'fadein'    : self.fadein['D'],
                'phase'     : self.phase,
                'complete': self.complete,
                'flag_flush': self.flag_flush,
            }
            return state


    def save_ckpt(self, path):

        if not os.path.exists(path):
            os.makedirs(path)
        ndis = 'dis_E{}_I{}_R{}x{}_{}.pth.tar'.format(self.epoch, self.globalIter, self.img_size, self.img_size, self.phase)
        ngen = 'gen_E{}_I{}_R{}x{}_{}.pth.tar'.format(self.epoch, self.globalIter, self.img_size, self.img_size, self.phase)
        save_path = os.path.join(path, ndis)
        if not os.path.exists(save_path):
            # ship models to cpu in get_state and save models
            torch.save(self.get_state('D'), save_path)
            save_path = os.path.join(path, ngen)
            torch.save(self.get_state('G'), save_path)
            print(' ... saving model checkpoints to {}'.format(path))

        # re-ship everything to cuda
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()


    def get_final_state(self, target):

        # ship models to cpu
        self.G_save = self.G.cpu()
        self.D_save = self.D.cpu()

        if target == 'G':
            state = {
                'G_structure': self.G_save,
                'resl' : self.resl,
                'state_dict' : self.G_save.state_dict(),
            }
            return state

        elif target == 'D':
            state = {
                'D_structure': self.D_save,
                'resl' : self.resl,
                'state_dict' : self.D_save.state_dict(),
            }
            return state


    def save_final_model(self, path):

        if not os.path.exists(path):
            os.makedirs(path)
        ndis = 'dis_E{}_I{}_R{}x{}_final.pth.tar'.format(self.epoch, self.globalIter, self.img_size, self.img_size)
        ngen = 'gen_E{}_I{}_R{}x{}_final.pth.tar'.format(self.epoch, self.globalIter, self.img_size, self.img_size)
        save_path = os.path.join(path, ndis)
        if not os.path.exists(save_path):
            # ship models to cpu in get_state and save models
            torch.save(self.get_final_state('D'), save_path)
            save_path = os.path.join(path, ngen)
            torch.save(self.get_final_state('G'), save_path)
            print(' ... saving final models to {}'.format(path))


    def var2np(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()


# use cudnn backends to boost speed
torch.backends.cudnn.benchmark = True

if config.data_root=='':
    raise Exception('Path to training data is undefined! Please specify the path in the --data_root flag!')
else:
    trainer = Trainer(config)
    trainer.train()
