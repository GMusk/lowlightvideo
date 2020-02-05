import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
import argparse

# add argument parser for clean use
parser = argparse.ArgumentParser(description="Video Enhancer")
parser.add_argument('-p', '--path', required=True, type=str, help='source of video file')
parser.add_argument('-l', '--loop', action='store_true', help='repeat loop of video')
parser.add_argument('-s', '--size', nargs=2, type=int, help='size of video to display')
parser.add_argument('-w', '--write', action='store_true', help='store processed video')
args = parser.parse_args()


class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = []
        self.bufferSize = 30

    def add(self, frame):
        # check buffer isnt full
        averageFrame = None
        if len(self.queue) >= self.bufferSize:
            averageFrame = self.average()
            self.queue.pop(0)
        self.queue.append(frame)
        return averageFrame

    def average(self):
        average = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint16)
        for frame in self.queue:
            average += frame
        return average / self.bufferSize


class FrameEditor:
    def __init__(self, gamma=1.5):
        self.gamma = gamma
        # get lookup table
        self.lookup = np.array([math.log(i, 10) * 106 for i in np.arange(1, 257)], dtype=np.uint8)

    def doOperation(self, frame):
        # gamma correction
        gamma_corrected_f = np.power(frame/ 255, self.gamma) * 255
        # check for overflow
        if gamma_corrected_f.max() > 255:
            gamma_corrected_f = gamma_corrected_f * (255 / gamma_corrected_f.max())
        gamma_corrected_i = gamma_corrected_f.astype("uint8")
        # apply lookup table
        edited = cv2.LUT(gamma_corrected_i, self.lookup)
        # return gamme
        gamma_final_f = np.power(edited / 255, 1 / 2.2) * 255
        if gamma_final_f.max() > 255:
            gamma_final_f = gamma_final_f * (255 / gamma_final_f.max())
        gamma_final_i = gamma_final_f.astype("uint8")
        return gamma_final_i


class ObjectDetector:
    def __init__(self):
        pass

    def getMoving(frame):
        return


def getVideo(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("No Video Found")
        return None
    else:
        return cap


def loopVideo(cap, size, loop, vw):
    # initiate average and edit class
    ma = MovingAverage(size)
    fe = FrameEditor()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # get moving objects
            movingObjects = getMoving(frame)
            # Call image operations here
            editFrame = fe.doOperation(frame)

            # store frame in moving average
            av_frame = ma.add(editFrame)

            # smoothFrame = ma.get_current() / 255.0

            if vw and av_frame is not None:
                # resize
                # r_frame = cv2.resize(av_frame, (size[0], size[1]))
                av_frame = np.uint8(av_frame)
                vw.write(av_frame)
        else:
            # if looping
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print("video finished")
                break
        # key to break playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def histogram():
    # get each color plane
    bgr_planes = cv2.split(frame)

    # histogram calculation
    #                     source     , index, mask? , size, range
    b_hist = cv2.calcHist(bgr_planes, [0], None, [256], [0, 256])
    g_hist = cv2.calcHist(bgr_planes, [1], None, [256], [0, 256])
    r_hist = cv2.calcHist(bgr_planes, [2], None, [256], [0, 256])
    l_hist = cv2.calcHist([luma], [0], None, [256], [0, 256])

    # hist setup
    hist_w = int(512 * 0.75)
    hist_h = int(400 * 0.75)
    bin_w = int(round(hist_w / 256))

    histImageRGB = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histImageLum = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)
    cv2.normalize(l_hist, l_hist, alpha=0, beta=hist_h,
                  norm_type=cv2.NORM_MINMAX)

    for i in range(1, 256):
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(b_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(b_hist[i]))),
                 (255, 0, 0), thickness=2)
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(g_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(g_hist[i]))),
                 (0, 255, 0), thickness=2)
        cv2.line(histImageRGB,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(r_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(r_hist[i]))),
                 (0, 0, 255), thickness=2)
        cv2.line(histImageLum,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(l_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(l_hist[i]))),
                 (255, 255, 255), thickness=2)

    # view luma as 3 channels to allow concatenation
    luma_3_channel = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate((frame, luma_3_channel), axis=1)

    # Display the resulting frame
    cv2.imshow('frame', combined)
    cv2.imshow('histrgb', histImageRGB)
    cv2.imshow('histluma', histImageLum)


def equal():
    # equalise histogram
    eq = cv2.equalizeHist(luma)
    # get hist
    l_hist = cv2.calcHist([eq], [0], None, [256], [0, 256])

    # hist setup
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))

    histImageLum = np.zeros((hist_h, hist_w, 1), dtype=np.uint8)

    for i in range(1, 256):
        cv2.line(histImageLum,
                 (bin_w * (i - 1),
                  hist_h - int(np.round(l_hist[i - 1]))),
                 (bin_w * (i), hist_h - int(np.round(l_hist[i]))),
                 (255, 255, 255), thickness=2)

    combined = np.concatenate((luma, eq), axis=1)
    cv2.imshow('frame', combined)
    cv2.imshow('histluma', histImageLum)


def clahe():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(luma)
    combined = np.concatenate((luma, cl1), axis=1)
    cv2.imshow('frame', combined)


def main(args):
    # get capture object
    cap = getVideo(args.path)
    videoWriter = None
    if args.size != None:
        size = tuple(args.size)
    else:
        # get caps original size
        size = (int(cap.get(3)), int(cap.get(4)))
    if args.write == True:
        # set codec for written video
        filename = args.path.split('.')[0] + '-out' + '.mp4'
        print(filename)
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter(filename, fourcc, fps, size)
    if cap != None:
        loopVideo(cap, size, args.loop, videoWriter)
    # When everything done, release the capture
    if videoWriter is not None:
        videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # when run as script
    main(args)
