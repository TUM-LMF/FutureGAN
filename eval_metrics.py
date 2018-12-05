# =============================================================================
# Evaluation Metrics
# =============================================================================

import math
import numpy as np
from scipy import signal, linalg
from scipy.stats import entropy
from skimage.measure import compare_psnr, compare_ssim, compare_mse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3


# =============================================================================
# calculate MSE

def calculate_mse(img_gen, img_gt):
    '''
    Computes the Mean Squared Error the generated images and the ground
    truth images.
    '''

    img_gen = img_gen.permute(1,2,0)
    img_gt = img_gt.permute(1,2,0)

    img_gen = img_gen.cpu()
    img_gen = img_gen.data.numpy()
    img_gt = img_gt.cpu()
    img_gt = img_gt.data.numpy()

    mse = compare_mse(img_gt, img_gen)
    return mse


# =============================================================================
# calculate PSNR

def calculate_psnr(img_gen, img_gt):
    '''
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.
    '''

    img_gen = img_gen.permute(1,2,0)
    img_gt = img_gt.permute(1,2,0)

    img_gen = img_gen.data.cpu().numpy()
    img_gt = img_gt.data.cpu().numpy()

    psnr = compare_psnr(img_gt, img_gen)
    return psnr


# =============================================================================
# calculate SSIM

def calculate_ssim(img_gen, img_gt):
    '''
    Returns the Structural Similarity Map between `img_gen` and `img_gt`.
    '''

    img_gen = img_gen.permute(1,2,0)
    img_gt = img_gt.permute(1,2,0)

    img_gen = img_gen.data.cpu().numpy()
    img_gt = img_gt.data.cpu().numpy()

    ssim = compare_ssim(img_gen, img_gt, multichannel=True)
    return ssim


# =============================================================================
# calculate SSIM and contrast sensitivity (similar to Oliu et al. ECCV 2018)
# -----------------------------------------------------------------------------
# code reference: https://github.com/tkarras/progressive_growing_of_gans

def fspecial_gauss(size, sigma):
    '''
    Function to mimic the 'fspecial' gaussian MATLAB function.
    '''

    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    return g / g.sum()


def calculate_ssim2(img_gen, img_gt, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, mssim=''):
    '''
    Return the Structural Similarity Map between `img_gen` and `img_gt`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img_gen: Numpy array holding the first RGB image.
        img_gt: Numpy array holding the second RGB image.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).

    Returns:
        Pair containing the mean SSIM and contrast sensitivity between `img_gen` and
        `img_gt`.
    '''
    if mssim=='':
        img_gen = img_gen.data.cpu().numpy()
        img_gt = img_gt.data.cpu().numpy()

    _, height, width = img_gen.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(fspecial_gauss(size, sigma), (1, size, size))
        mu1 = signal.fftconvolve(img_gen, window, mode='valid')
        mu2 = signal.fftconvolve(img_gt, window, mode='valid')
        sigma11 = signal.fftconvolve(img_gen * img_gen, window, mode='valid')
        sigma22 = signal.fftconvolve(img_gt * img_gt, window, mode='valid')
        sigma12 = signal.fftconvolve(img_gen * img_gt, window, mode='valid')

    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img_gen, img_gt
        sigma11 = img_gen * img_gen
        sigma22 = img_gt * img_gt
        sigma12 = img_gen * img_gt

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = k1 ** 2
    c2 = k2  ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)), axis=(0, 1, 2)) # Return for each image individually.
#    cs = np.mean(v1 / v2, axis=(0, 1, 2))
    return ssim#, cs


# =============================================================================
# calculate MS-SSIM
# -----------------------------------------------------------------------------
# code reference: https://github.com/tkarras/progressive_growing_of_gans

def hox_downsample(img):
    return (img[:, 0::2, 0::2] + img[:, 1::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 1::2]) * 0.25


def calculate_ms_ssim(img_gen, img_gt, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    '''
    Return the MS-SSIM score between generated and ground truth images.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img_gen: Numpy array holding the generated RGB image.
        img_gt: Numpy array holding the ground truth RGB image.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).
        weights: List of weights for each level; if none, use five levels and the
            weights from the original paper.

    Returns:
        MS-SSIM score between `img_gen` and `img_gt`.

    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [depth, height, width].
    '''

    img_gen = img_gen.data.cpu().numpy()
    img_gt = img_gt.data.cpu().numpy()

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [img_gen, img_gt]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = calculate_ssim2(im1, im2, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2, mssim='mssim')
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    # Average over images only at the end.
    return np.mean(np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]))
