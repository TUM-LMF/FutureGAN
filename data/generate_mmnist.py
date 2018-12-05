# this code is based on

from __future__ import absolute_import, division, print_function

from tqdm import tqdm

import cv2
import mnist
import itertools
import math
import numpy as np
import os

from PIL import Image


def load_dataset(mode = "train"):
    print("Loading Mnist {}".format(mode))
    if mode == 'train':
        mnist_images = mnist.train_images()
        mnist_labels = mnist.train_labels()
    elif mode == 'test':
        mnist_images = mnist.test_images()
        mnist_labels = mnist.test_labels()

    n_labels = np.unique(mnist_labels)
    mnist_dict = {}
    mnist_sizes = []
    for i in n_labels:
        idxs = np.where(mnist_labels == i)
        mnist_dict[i] = mnist_images[idxs]
        mnist_sizes.append(mnist_dict[i].shape[0])

    return mnist_dict, mnist_sizes


# helper functions
def arr_from_img(im, shift=0):
    w, h = im.size
    arr = im.getdata()
    c = np.product(arr.size) // (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h, w, c)).transpose(2, 1, 0) / 255. - shift


def get_random_images(dataset, size_list, id_list):
    images = []
    for id in id_list:
        idx = np.random.randint(0, size_list[id])
        images.append(dataset[id][idx])

    return images


def generate_moving_mnist(shape=(64, 64), seq_len=30, seqs_per_class=100, num_sz=28, nums_per_image=2, mode="train"):
    mnist_dict, mnist_sizes = load_dataset(mode)
    combinations = list(itertools.combinations(list(mnist_dict.keys()), nums_per_image))
    comb_len = len(combinations)

    width, height = shape
    lims = (x_lim, y_lim) = width-num_sz, height-num_sz


    print("Generating Moving Mnist: {}".format(mode))
    moving_mnist = {}
    loop = tqdm(combinations, total=comb_len)
    for combination in loop:
        moving_mnist[combination] = []
        for seq_idx in range(seqs_per_class):
            # randomly generate direc/speed/position, calculate velocity vector
            direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
            speeds = np.random.randint(5, size=nums_per_image)+2
            veloc = [(v*math.cos(d), v*math.sin(d)) for d, v in zip(direcs, speeds)]
            positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in range(nums_per_image)]

            mnist_images = get_random_images(mnist_dict, mnist_sizes, combination)
            video = []

            for frame_idx in range(seq_len):
                canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
                canvas = np.zeros((1, width, height), dtype=np.float32)

                for i, canv in enumerate(canvases):
                    canv.paste(Image.fromarray(mnist_images[i]), tuple(map(lambda p: int(round(p)), positions[i])))
                    canvas += arr_from_img(canv, shift=0)
                # update positions based on velocity
                next_pos = [map(sum, zip(p, v)) for p, v in zip(positions, veloc)]
                # bounce off wall if a we hit one
                for i, pos in enumerate(next_pos):
                    for j, coord in enumerate(pos):
                        if coord < -2 or coord > lims[j]+2:
                            veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))

                positions = [(p_[0] + v_[0], p_[1] + v_[1]) for p_, v_ in zip(positions, veloc)]

                image = (canvas * 255).astype(np.uint8).clip(0, 255).transpose(2, 1, 0)
                video.append(image)
                # image = np
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            moving_mnist[combination].append(video)

        moving_mnist[combination] = np.array(moving_mnist[combination])

    return moving_mnist


def get_comb_name(comb_list):
    name = ""
    for i in comb_list:
        name += str(i)
    return name


def save_dataset(dataset, filetype, nums_per_image, dataset_path, list_filename, mode="train"):
    if filetype == 'png':
        dataset_path = "{}-{}".format(dataset_path, nums_per_image)
        if not os.path.exists(dataset_path):
	          os.makedirs(dataset_path)
        list_filename = "{}/{}".format(dataset_path, list_filename)
        with open(list_filename, 'w') as writer:
            for combination, video_list in dataset.items():
                comb_name = get_comb_name(combination)
                comb_path = "{}/{}/{}".format(dataset_path, mode, comb_name)

                for i, video in enumerate(video_list):
                    video_path = "%s_%05d" % (comb_path, i)

                    writer.write("{} {}\n".format(video_path, comb_name))
                    if not os.path.exists(video_path):
            	          os.makedirs(video_path)
                    for j, image in enumerate(video):
                        output_filename = "%s/%s_%05d_%04d.png" % (video_path, comb_name, i, j)
                        cv2.imwrite(output_filename, image)


def main():

    dataset_path = "./mmnist"
    train_list_filename = "dataset_train.list"
    test_list_filename = "dataset_test.list"
    filetype = "png"
    width = 64
    height = 64
    seq_len = 36
    seqs_per_class = 120
    seqs_per_class_test = 50
    num_sz = 28
    nums_per_image = 2

    # Train

    moving_mnist = generate_moving_mnist(shape=(height, width), seq_len=seq_len, seqs_per_class=seqs_per_class, \
                                num_sz=num_sz, nums_per_image=nums_per_image)

    save_dataset(moving_mnist, filetype, nums_per_image, dataset_path, train_list_filename, mode="train")

    # Test

    moving_mnist = generate_moving_mnist(shape=(height, width), seq_len=seq_len, seqs_per_class=seqs_per_class_test, \
                                num_sz=num_sz, nums_per_image=nums_per_image, mode="test")

    save_dataset(moving_mnist, filetype, nums_per_image, dataset_path, test_list_filename, mode="test")

    print("Finish")

if __name__ == '__main__':
    main()
