# =============================================================================
# Test and Evaluate FutureGAN
# =============================================================================

'''
Script to
    1. generate and evaluate test samples of FutureGAN, or
    2. calculate evaluation metrics for existing test samples of a baseline model (`CopyLast`, `fRNN`, `MCNet`).

-------------------------------------------------------------------------------
1. To generate and evaluate test samples of FutureGAN, please set the --model_path flag correctly:
    --model_path=`path_to_FutureGAN_generator_ckpt`

Your test data to generate predictions and evaluate FutureGAN is assumed to be arranged in this way:
    data_root/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.

For evaluation you can choose which metrics are calculated, please set the --metrics flag accordingly.
Your choices are: `mse`, `psnr`, `ssim`, `ssim2`, `ms_ssim`.
If you want to calculate multiple metrics, simply append them using the --metrics flag:
    --metrics=`metric1` --metrics=`metric2` ...

-------------------------------------------------------------------------------
2. To calculate evaluation metrics for existing test samples of a baseline model, please set the --model flag correctly:
    --baseline=`shortname_of_baseline_model`, one of: `CopyLast`, `fRNN`, `MCNet`

Your data to evaluate the results of `CopyLast` baseline is assumed to be arranged in this way:
    data_root/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.

Your data to evaluate the results of a baseline model other than `CopyLast` is assumed to be arranged in this way:
    Ground truth frames:
        data_root/in_gt/video(n)/frame(m).ext
    Predicted frames
        data_root/in_pred/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.

For evaluation you can choose which metrics are calculated, please set the --metrics flag accordingly.
Your choices are: `mse`, `psnr`, `ssim`, `ssim2`, `ms_ssim`.
If you want to calculate multiple metrics, simply append them using the --metrics flag:
    --metrics=`metric1` --metrics=`metric2` ...

-------------------------------------------------------------------------------
For further options and information, read the provided `help` information of the optional arguments below.
'''

import os
import time
import argparse
from PIL import Image
from math import floor, ceil
import numpy as np
import imageio
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils import get_image_grid, save_image_grid, count_model_params
import eval_metrics as eval_metrics
from video_dataset import VideoFolder, video_loader
from torch.utils.data import DataLoader, sampler


# =============================================================================
# config options

help_description = 'This script evaluates a FutureGAN model or one of these baseline models: `CopyLast` `fRNN` `MCNet`, according to the specified arguments.'

parser = argparse.ArgumentParser(description=help_description)

# general
parser.add_argument('--random_seed', type=int, default=int(time.time()), help='seed for generating random numbers, default = `int(time.time())`')
parser.add_argument('--ext', action='append', default=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'], help='list of strings of allowed file extensions, default=[.jpg, .jpeg, .png, .ppm, .bmp, .pgm]')

parser.add_argument('--model_path', type=str, default='', help='path to FutureGAN`s generator checkpoint, default=``')
parser.add_argument('--model', type=str, default='FutureGAN', help='model to calculate evaluation metrics for (choices: `FutureGAN`, `CopyLast`, `MCNet`, `fRNN`), default=`FutureGAN`')
parser.add_argument('--data_root', type=str, default='', help='path to root directory of test data (ex. -->path_to_dataset/test)')
parser.add_argument('--test_dir', type=str, default='./tests', help='path to directory for saving test results, default=`./tests`')
parser.add_argument('--experiment_name', type=str, default='', help='name of experiment, default=``')

parser.add_argument('--nc', type=int, default=3, help='number of input image channels, default=3')
parser.add_argument('--resl', type=int, default=128, help='frame resolution, default=128')
parser.add_argument('--nframes_pred', type=int, default=6, help='number of video frames to generate or predict for one sample, default=6')
parser.add_argument('--nframes_in', type=int, default=6, help='number of video frames in one sample, default=6')
parser.add_argument('--deep_pred', type=int, default=1, help='number of (recursive) prediction steps for future generator in test mode, default=1')
parser.add_argument('--batch_size', type=int, default=8, help='batch size at test time, change according to available gpu memory, default=8')
parser.add_argument('--metrics', action='append', help='list of evaluation metrics to calculate (choices: `mse`, `psnr`, `ssim`, `ssim2`), default=``')

# display and save
parser.add_argument('--save_frames_every', type=int, default=1, help='save video frames every specified iteration, default=1')
parser.add_argument('--save_gif_every', type=int, default=1, help='save gif every specified iteration, default=1')
parser.add_argument('--in_border', type=str, default='black', help='color of border added to gif`s input frames (`color_name`), default=`black`')
parser.add_argument('--out_border', type=str, default='red', help='color of border added to gif`s output frames (`color_name`), default=`red`')
parser.add_argument('--npx_border', type=int, default=2, help='number of border pixels, default=2')

# parse and save training config
config = parser.parse_args()


# =============================================================================
# enable cuda if gpu(s) is/are available

if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')


# =============================================================================
# evaluation routine for video prediction

def evaluate_pred(config):

    # define directories
    model_name=config.model

    test_data_root = config.data_root
    if config.deep_pred>1:
        test_dir = config.test_dir+'/'+config.experiment_name+'/deep-pred{}/'.format(config.deep_pred)+model_name
    else:
        test_dir = config.test_dir+'/'+config.experiment_name+'/pred/'+model_name
    if not os.path.exists(test_dir):
            os.makedirs(test_dir)
    sample_dir = test_dir+'/samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    nframes_in = config.nframes_in
    nframes_pred = config.nframes_pred*config.deep_pred
    nframes = nframes_in+nframes_pred
    img_size = int(config.resl)
    nworkers = 4

    # load model
    if config.model=='FutureGAN':
        ckpt = torch.load(config.model_path)
        # model structure
        G = ckpt['G_structure']
        # load model parameters
        G.load_state_dict(ckpt['state_dict'])
        G.eval()
        G = G.module.model
        if use_cuda:
            G = G.cuda()
        print(' ... loading FutureGAN`s FutureGenerator from checkpoint: {}'.format(config.model_path))

    # load test dataset
    transform = transforms.Compose([transforms.Resize(size=(img_size,img_size), interpolation=Image.NEAREST), transforms.ToTensor(),])
    if config.model=='FutureGAN' or config.model=='CopyLast':
        dataset_gt = VideoFolder(video_root=test_data_root, video_ext=config.ext, nframes=nframes, loader=video_loader, transform=transform)
        dataloader_gt = DataLoader(dataset=dataset_gt, batch_size=config.batch_size, sampler=sampler.SequentialSampler(dataset_gt), num_workers=nworkers)
    else:
        dataset_gt = VideoFolder(video_root=test_data_root+'/in_gt', video_ext=config.ext, nframes=nframes, loader=video_loader, transform=transform)
        dataset_pred = VideoFolder(video_root=test_data_root+'/in_pred', video_ext=config.ext, nframes=nframes, loader=video_loader, transform=transform)
        dataloader_pred = DataLoader(dataset=dataset_pred, batch_size=config.batch_size, sampler=sampler.SequentialSampler(dataset_pred), num_workers=nworkers)
        dataloader_gt = DataLoader(dataset=dataset_gt, batch_size=config.batch_size, sampler=sampler.SequentialSampler(dataset_gt), num_workers=nworkers)
        data_iter_pred = iter(dataloader_pred)
    test_len = len(dataset_gt)
    data_iter_gt = iter(dataloader_gt)

    # save model structure to file
    if config.model=='FutureGAN':
        # count model parameters
        nparams_g = count_model_params(G)
        with open(test_dir+'/model_structure_{}x{}.txt'.format(img_size, img_size),'w') as f:
            print('--------------------------------------------------', file=f)
            print('Sequences in test dataset: ', len(dataset_gt), file=f)
            print('Number of model parameters: ', file=f)
            print(nparams_g, file=f)
            print('--------------------------------------------------', file=f)
            print('Model structure: ', file=f)
            print(G, file=f)
            print('--------------------------------------------------', file=f)
            print(' ... FutureGAN`s FutureGenerator has been loaded successfully from checkpoint ... ')
            print(' ... saving model struture to {}'.format(f))

    # save test configuration
    with open(test_dir+'/eval_config.txt','w') as f:
        print('------------- test configuration -------------', file=f)
        for l, m in vars(config).items():
            print(('{}: {}').format(l, m), file=f)
        print(' ... loading test configuration ... ')
        print(' ... saving test configuration {}'.format(f))

    # define tensors
    if config.model=='FutureGAN':
        print(' ... testing FutureGAN ...')
        if config.deep_pred>1:
            print(' ... recursively predicting {}x{} future frames from {} input frames ...'.format(config.deep_pred, config.nframes_pred, nframes_in))
        else:
            print(' ... predicting {} future frames from {} input frames ...'.format(nframes_pred, nframes_in))
    z = Variable(torch.FloatTensor(config.batch_size, config.nc, nframes_in, img_size, img_size))
    z_in = Variable(torch.FloatTensor(config.batch_size, config.nc, nframes_in, img_size, img_size))
    x_pred = Variable(torch.FloatTensor(config.batch_size, config.nc, nframes_pred, img_size, img_size))
    x = Variable(torch.FloatTensor(config.batch_size, config.nc, nframes, img_size, img_size))
    x_eval = Variable(torch.FloatTensor(config.batch_size, config.nc, nframes_pred, img_size, img_size))

    # define tensors for evaluation
    if config.metrics is not None:
        print(' ... evaluating {} ...'.format(model_name))
        if 'ms_ssim' in config.metrics and img_size<32:
            raise ValueError('For calculating `ms_ssim`, your dataset must consist of images at least of size 32x32!')

        metrics_values = {}
        for metric_name in config.metrics:
            metrics_values['{}_frames'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(test_len, nframes_pred))
            metrics_values['{}_avg'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(test_len,1))
            print(' ... calculating {} ...'.format(metric_name))

    # test loop
    if config.metrics is not None:
        metrics_i_video = {}
        for metric_name in config.metrics:
            metrics_i_video['{}_i_video'.format(metric_name)] = 0

    i_save_video = 1
    i_save_gif = 1

    for step in tqdm(range(len(data_iter_gt))):

        # input frames
        x.data = next(data_iter_gt)
        x_eval.data = x.data[:,:,nframes_in:,:,:]
        z.data = x.data[:,:,:nframes_in,:,:]

        if use_cuda:
            x = x.cuda()
            x_eval = x_eval.cuda()
            z = z.cuda()
            x_pred = x_pred.cuda()

        # predict video frames
        # !!! TODO !!! for deep_pred > 1: correctly implemented only if nframes_in == nframes_pred
        if config.model=='FutureGAN':
            z_in.data = z.data
            for i_deep_pred in range(0,config.deep_pred):
                x_pred[:z_in.size(0),:,i_deep_pred*config.nframes_pred:(i_deep_pred*config.nframes_pred)+config.nframes_pred,:,:] = G(z_in).detach()
                z_in.data = x_pred.data[:,:,i_deep_pred*config.nframes_pred:(i_deep_pred*config.nframes_pred)+config.nframes_pred,:,:]

        elif config.model=='CopyLast':
            for i_baseline_frame in range(x_pred.size(2)):
                x_pred.data[:x.size(0),:,i_baseline_frame,:,:] = x.data[:,:,nframes_in-1,:,:]

        else:
            x_pred.data = next(data_iter_pred)[:x.size(0),:,nframes_in:,:,:]

        # calculate eval statistics
        if config.metrics is not None:
            for metric_name in config.metrics:
                calculate_metric = getattr(eval_metrics, 'calculate_{}'.format(metric_name))

                for i_batch in range(x.size(0)):
                    for i_frame in range(nframes_pred):
                        metrics_values['{}_frames'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)], i_frame] = calculate_metric(x_pred[i_batch,:,i_frame,:,:], x_eval[i_batch,:,i_frame,:,:])
                        metrics_values['{}_avg'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)]] = torch.mean(metrics_values['{}_frames'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)]])
                    metrics_i_video['{}_i_video'.format(metric_name)] = metrics_i_video['{}_i_video'.format(metric_name)]+1

        # save frames
        if config.save_frames_every is not 0 and config.model=='FutureGAN':
            if step%config.save_frames_every==0 or step==0:
                for i_save_batch in range(x.size(0)):
                    if not os.path.exists(sample_dir+'/in_gt/video{:04d}'.format(i_save_video)):
                        os.makedirs(sample_dir+'/in_gt/video{:04d}'.format(i_save_video))
                    if not os.path.exists(sample_dir+'/in_pred/video{:04d}'.format(i_save_video)):
                        os.makedirs(sample_dir+'/in_pred/video{:04d}'.format(i_save_video))
                    for i_save_z in range(z.size(2)):
                        save_image_grid(z.data[i_save_batch,:,i_save_z,:,:].unsqueeze(0), sample_dir+'/in_gt/video{:04d}/video{:04d}_frame{:04d}_R{}x{}.png'.format(i_save_video, i_save_video, i_save_z+1, img_size, img_size), img_size, 1)
                        save_image_grid(z.data[i_save_batch,:,i_save_z,:,:].unsqueeze(0), sample_dir+'/in_pred/video{:04d}/video{:04d}_frame{:04d}_R{}x{}.png'.format(i_save_video, i_save_video, i_save_z+1, img_size, img_size), img_size, 1)
                    for i_save_x_pred in range(x_pred.size(2)):
                        save_image_grid(x_eval.data[i_save_batch,:,i_save_x_pred,:,:].unsqueeze(0), sample_dir+'/in_gt/video{:04d}/video{:04d}_frame{:04d}_R{}x{}.png'.format(i_save_video, i_save_video, i_save_x_pred+1+nframes_in, img_size, img_size), img_size, 1)
                        save_image_grid(x_pred.data[i_save_batch,:,i_save_x_pred,:,:].unsqueeze(0), sample_dir+'/in_pred/video{:04d}/video{:04d}_frame{:04d}_R{}x{}.png'.format(i_save_video, i_save_video, i_save_x_pred+1+nframes_in, img_size, img_size), img_size, 1)
                    i_save_video = i_save_video+1

        # save gifs
        if config.save_gif_every is not 0:
            if step%config.save_gif_every==0 or step==0:
                for i_save_batch in range(x.size(0)):
                    if not os.path.exists(sample_dir+'/in_gt/video{:04d}'.format(i_save_gif)):
                        os.makedirs(sample_dir+'/in_gt/video{:04d}'.format(i_save_gif))
                    if not os.path.exists(sample_dir+'/in_pred/video{:04d}'.format(i_save_gif)):
                        os.makedirs(sample_dir+'/in_pred/video{:04d}'.format(i_save_gif))
                    frames = []
                    for i_save_z in range(z.size(2)):
                        frames.append(get_image_grid(z.data[i_save_batch,:,i_save_z,:,:].unsqueeze(0), img_size, 1, config.in_border, config.npx_border))
                    for i_save_x_pred in range(x_pred.size(2)):
                        frames.append(get_image_grid(x_eval.data[i_save_batch,:,i_save_x_pred,:,:].unsqueeze(0), img_size, 1, config.out_border, config.npx_border))
                    imageio.mimsave(sample_dir+'/in_gt/video{:04d}/video{:04d}_R{}x{}.gif'.format(i_save_gif, i_save_gif, img_size, img_size), frames)
                    frames = []
                    for i_save_z in range(z.size(2)):
                        frames.append(get_image_grid(z.data[i_save_batch,:,i_save_z,:,:].unsqueeze(0), img_size, 1, config.in_border, config.npx_border))
                    for i_save_x_pred in range(x_pred.size(2)):
                        frames.append(get_image_grid(x_pred.data[i_save_batch,:,i_save_x_pred,:,:].unsqueeze(0), img_size, 1, config.out_border, config.npx_border))
                    imageio.mimsave(sample_dir+'/in_pred/video{:04d}/video{:04d}_R{}x{}.gif'.format(i_save_gif, i_save_gif, img_size, img_size), frames)
                    i_save_gif = i_save_gif+1

    if config.save_frames_every is not 0 and config.model=='FutureGAN':
        print(' ... saving video frames to dir: {}'.format(sample_dir))
        if config.save_gif_every is not 0:
            print(' ... saving gifs to dir: {}'.format(sample_dir))

    # calculate and save mean eval statistics
    if config.metrics is not None:
        metrics_mean_values = {}
        for metric_name in config.metrics:
            metrics_mean_values['{}_frames'.format(metric_name)] = torch.mean(metrics_values['{}_frames'.format(metric_name)],0)
            metrics_mean_values['{}_avg'.format(metric_name)] = torch.mean(metrics_values['{}_avg'.format(metric_name)],0)
            torch.save(metrics_mean_values['{}_frames'.format(metric_name)], os.path.join(test_dir, '{}_frames.pt'.format(metric_name)))
            torch.save(metrics_mean_values['{}_avg'.format(metric_name)], os.path.join(test_dir, '{}_avg.pt'.format(metric_name)))

        print(' ... saving evaluation statistics to dir: {}'.format(test_dir))


if config.model_path=='' and config.model=='FutureGAN':
    raise Exception('Path to model checkpoint is undefined! Please set --model_path flag or set --baseline flag to define a baseline to be evaluated (choices are: `CopyLast`, `fRNN`, `MCNet`)!')
elif config.deep_pred>1 and config.nframes_pred is not config.nframes_in:
    raise Exception('For recursive prediction, the number of input and predicted frames (per prediction step) must be equal!')
else:
    evaluate_pred(config)
