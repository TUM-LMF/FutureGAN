# =================================================================================================
# Image Preprocessing: Extracting and Resizing (bicubic) Frames from Video to unique Video Folders
# =================================================================================================

import os, subprocess
import glob
from PIL import Image

data_dir = './kth'
vid_ext = 'avi'
frame_rate = 25
img_size = 128
save_dir = os.path.abspath('./kth'+'_resized_'+str(img_size)+'x'+str(img_size))
img_ext = 'png'

os.chdir(data_dir)

for n in range(17,26):
    test_split_persons = ['person{:02d}'.format(n)]

for i, file in enumerate(glob.glob('**/*.'+vid_ext, recursive=True)):
    folder_name, file_name = os.path.split(file)

    video_name = file_name.split('_')[1]+'_'+file_name.split('_')[0]+'_'+file_name.split('_')[2]

    if file_name.split('_')[0] in test_split_persons:
        split = 'test'
    else: 
        split = 'train'

    frame_path_name = save_dir+'/'+split+'/'+video_name

    if not os.path.exists(frame_path_name):
        os.makedirs(frame_path_name)

    subprocess.call(['ffmpeg', '-i', '{}'.format(os.path.abspath(file)), '-r', '{}'.format(frame_rate), '-s', '{}x{}'.format(img_size, img_size), '{}/{}_%04d.{}'.format(frame_path_name, video_name, img_ext)])

    if i % 100==0:
        print('extracting frames from video', str(i+1), ':', file_name)
        print('... from video directory:', folder_name)
        print('... to new frame directory:', frame_path_name)

print('All video frames have been extracted and resized to unique video folders successfully!')


# =============================================================================
# Image Preprocessing: Converting RGB Frames to Grayscale Frames
# =============================================================================

os.chdir(save_dir)

for i, file in enumerate(sorted(glob.glob('**/*.'+img_ext, recursive=True))):
    folder_name, file_name = os.path.split(file)

    img = Image.open(os.path.abspath(file)).convert('L')
    img.save(os.path.abspath(file))

    if i % 100==0:
        print('converting image', str(i+1), ':', file_name)
        print('... in directory:', folder_name)

print('All video frames have been converted from rgb to grayscale successfully!')
