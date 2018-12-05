# =============================================================================
# Plot Evaluation Results
# =============================================================================

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='', help='path to root directory of test results, default=``')
parser.add_argument('--experiment_name', type=str, default='', help='name of experiment, default=``')

parser.add_argument('--nframes_pred', type=int, default=6, help='number of predicted frames')
parser.add_argument('--deep_pred', type=int, default=1, help='number of (recursive) prediction steps, default=1')

parser.add_argument('--models', action='append', default=['FutureGAN', 'CopyLast'], help='list of evaluated models (choices: `FutureGAN`, `CopyLast`, `fRNN`, `MCNet`), default=[]`FutureGAN`, `CopyLast`]')
parser.add_argument('--metrics', action='append', default=['mse', 'psnr', 'ssim'], help='list of calculated evaluation metrics (choices: `mse`, `psnr`, `ssim`, `ssim2`, `ms_ssim`), default=[`mse`, `psnr`, `ssim`]')

parser.add_argument('--fontsize', type=int, default=12, help='fontsize for plots in pt, default=12')
parser.add_argument('--fontfamily', type=str, default='serif', help='fontfamily for plots, default=`serif`')
parser.add_argument('--legend_location', type=int, default=1, help='location of legend in plot, default=1')

config = parser.parse_args()


# directories for loading and saving
if config.deep_pred>1:
    eval_data_root = config.data_root+'/'+config.experiment_name+'/deep-pred{}/'.format(config.deep_pred)
else:
    eval_data_root = config.data_root+'/'+config.experiment_name+'/pred/'

model_dirs = {}
for model_name in config.models:
    model_dirs['{}'.format(model_name)] = eval_data_root+model_name
save_dir = eval_data_root

# load saved eval values
model_metrics_values = {}
for model_name in config.models:
    for metric_name in config.metrics:
        if metric_name=='inception_score':
            model_metrics_values['{}_{}_x_frames'.format(model_name, metric_name)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_x_frames.pt'.format(metric_name))
            model_metrics_values['{}_{}_x_avg'.format(model_name, metric_name)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_x_avg.pt'.format(metric_name))
            model_metrics_values['{}_{}_x_{}_frames'.format(model_name, metric_name, config.mode)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_x_pred_frames.pt'.format(metric_name))
            model_metrics_values['{}_{}_x_{}_avg'.format(model_name, metric_name, config.mode)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_x_pred_avg.pt'.format(metric_name))
        else:
            model_metrics_values['{}_{}_frames'.format(model_name, metric_name)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_frames.pt'.format(metric_name))
            model_metrics_values['{}_{}_avg'.format(model_name, metric_name)] = torch.load(model_dirs['{}'.format(model_name)]+'/{}_avg.pt'.format(metric_name))

# plot eval values
metric_label = {}
metric_label['mse'] = 'MSE'
metric_label['psnr'] = 'PSNR'
metric_label['ssim'] = 'SSIM'
metric_label['ssim2'] = 'SSIM'
metric_label['ms_ssim'] = 'MS-SSIM'

nframes = range(1,config.nframes_pred*config.deep_pred+1)

metrics_figure = {}
model_metrics_plots = {}
for metric_name in config.metrics:
    metrics_figure['{}_plot'.format(metric_name)] = plt.figure()
    ax = metrics_figure['{}_plot'.format(metric_name)].add_subplot(1, 1, 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for model_name in config.models:
        model_metrics_plots['{}_{}_plot'.format(model_name, metric_name)] = plt.plot(nframes, model_metrics_values['{}_{}_frames'.format(model_name, metric_name)].numpy(), label=model_name)
    plt.legend(loc = config.legend_location, prop={'size': config.fontsize-1, 'family': config.fontfamily})
    plt.xlabel('Number of Predicted Frame', fontsize=config.fontsize, family=config.fontfamily)
    plt.ylabel(metric_label['{}'.format(metric_name)], fontsize=config.fontsize, family=config.fontfamily)
    plt.grid(True, linestyle='dashed')
    plt.show()
    metrics_figure['{}_plot'.format(metric_name)].savefig(save_dir+'/{}.pdf'.format(metric_name), bbox_inches='tight')

# save eval average values to file
with open(save_dir+'/avg_eval_values.txt','w') as f:
    for model_name in config.models:
        print('--------------------------------------------------', file=f)
        for metric_name in config.metrics:
            print('{} {}: '.format(model_name, metric_label['{}'.format(metric_name)]), model_metrics_values['{}_{}_avg'.format(model_name, metric_name)], file=f)

# save eval per frame values to file
with open(save_dir+'/frame_eval_values.txt','w') as f:
    for model_name in config.models:
        print('--------------------------------------------------', file=f)
        for metric_name in config.metrics:
            print('{} {}: '.format(model_name, metric_label['{}'.format(metric_name)]), model_metrics_values['{}_{}_frames'.format(model_name, metric_name)], file=f)
