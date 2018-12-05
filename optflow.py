# =============================================================================
# Generate Optical Flow Maps
# =============================================================================

import numpy as np
import cv2
import itertools

# any cvVideoCap input (avi videos, webcams, image series, ...)
sequences = ['sequence0001', 'sequence0002', 'sequence0003', 'sequence0004', 'sequence0005', 'sequence0006']
models = ['FutureGAN', 'GroundTruth', 'MCNet', 'fRNN']
prefix = '/MovingMNIST/prediction_6frames/'
ext = '.png'

for sequence in sequences:
    for model in models:
        gif_name = sequence+'_'+model
        path = './test_data'+prefix
        filepath = path+sequence+'/'+ model
        videoname = path+gif_name

        # global flags
        blend = True
        repeat = True
        write = True
        printlegend = True

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        filename = filepath+'/'+sequence+'_frame%04d_R128x128'+ext

        cap = cv2.VideoCapture(filename)

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        if (write):
            writer = cv2.VideoWriter(videoname + '_OpticalFlow.avi', 0, 10, frame1.shape[0:2])
            repeat = False # don't repeat if you want to write a video


        if (printlegend):

            def makeLegend(xrange = range(-20,20)):
                xshift = np.min(xrange)
                yshift = np.min(xrange)

                hsv = np.zeros((np.size(xrange), np.size(xrange),3), dtype=np.uint8)
                hsv[..., 1] = 255

                for x,y in itertools.product(xrange, xrange):

                    if (np.linalg.norm((x,y)) <= np.ptp(xrange) / 2):
                        mag,ang = cv2.cartToPolar(x,y)
                        hsv[x + xshift, y + yshift, 0] = (ang[0] * 180) / (np.pi * 2)
                        hsv[x + xshift, y + yshift, 2] = (mag[0] / np.ptp(xrange)) * 255.

                return hsv

            legend = makeLegend(range(-10,10))

        i_write = 1
        while(True):
            ret, frame2 = cap.read()

            if (not ret):
                if (repeat):
                    cap.release()
                    cap = cv2.VideoCapture(filename)
                    continue
                else:
                    break

            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, next,
                                                None,
                                                .5, # pyramid scale
                                                5,  # levels
                                                25, # winsize
                                                15, # iterations
                                                7,  # polyN
                                                1.7,
                                                cv2.cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[..., 0] = (ang * 180) / (np.pi * 2)
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            if (printlegend):
                p = (np.shape(next)[0] - np.shape(legend)[0] - 5,
                     (np.shape(next)[1]- np.shape(legend)[1]) // 2)
                overlay = np.zeros_like(hsv)
                overlay[p[0]:p[0] + np.shape(legend)[0], p[1]:p[1] + np.shape(legend)[1]] = legend
                hsv[overlay[:,:,2] > 0] = overlay[overlay[:,:,2] > 0]

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            display = cv2.addWeighted(bgr, 1, frame2, 1, 0) if blend else bgr

            cv2.imshow('frame', display)

            if (cv2.waitKey(30) & 0xff) == 27:
                break

            if (writer.isOpened()):
                cv2.imwrite('{}/{}_OpticalFlow_frame{:04d}.png'.format(filepath, sequence, i_write), bgr)
                writer.write(display)

            prvs = next

            i_write += 1


        # tidy up
        cap.release()
        writer.release()
        #cv2.destroyAllWindows()
